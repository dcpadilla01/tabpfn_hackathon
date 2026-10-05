
import pandas as pd, numpy as np, agent_api

f = agent_api.load_saved('feats_v3.parquet')
print("feats_v3 shape:", f.shape)
print("cols:", list(f.columns))
print(f.head(3))

t = agent_api.train_targets()
print("\ntrain_targets:", t.shape)
print(t['future_spend_4w'].describe())
print("zero share:", (t['future_spend_4w']==0).mean())
print(t['snapshot_day'].value_counts().sort_index())
print("\nsnapshot_days:", agent_api.snapshot_days())


# ---- cell ----

import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
print("train rows:", tr.shape)

p6 = agent_api.load_saved('pred_e006.parquet')
p4 = agent_api.load_saved('pred_e004.parquet')
p5 = agent_api.load_saved('pred_e005.parquet')
for nm,p in [('e004',p4),('e005',p5),('e006',p6)]:
    print(nm, p.shape, list(p.columns))

# merge preds onto train rows (in-sample diagnostic)
m = tr[['household_key','snapshot_day','future_spend_4w','spend_28','spend_56','spend_84']].merge(
    p6.rename(columns={'prediction':'pred6'}), on=['household_key','snapshot_day'])
m = m.merge(p4.rename(columns={'prediction':'pred4'}), on=['household_key','snapshot_day'])
m = m.merge(p5.rename(columns={'prediction':'pred5'}), on=['household_key','snapshot_day'])
y = m['future_spend_4w']

def mae(a,b): return np.mean(np.abs(a-b))
print("\nIN-SAMPLE (train) diagnostics:")
print("MAE spend_28 naive:", mae(m['spend_28'], y))
print("MAE pred4:", mae(m['pred4'], y), "MAE pred5:", mae(m['pred5'], y), "MAE pred6:", mae(m['pred6'], y))
print("mean y:", y.mean(), "mean pred6:", m['pred6'].mean(), "median y:", y.median(), "median pred6:", m['pred6'].median())
print("corr(pred6,y):", np.corrcoef(m['pred6'], y)[0,1], " corr(spend28,y):", np.corrcoef(m['spend_28'], y)[0,1])

# bias by target decile
m['dec'] = pd.qcut(y, 10, duplicates='drop')
print(m.groupby('dec', observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'y_mean': g['future_spend_4w'].mean(), 'pred6_mean': g['pred6'].mean(),
    'bias': (g['pred6']-g['future_spend_4w']).mean()}), include_groups=False))

# how often pred6==0 vs y==0
print("\nzero pred6 share:", (m['pred6']<=1e-9).mean(), " zero y share:", (y==0).mean())
print("MAE on y==0 rows:", mae(m.loc[y==0,'pred6'], 0), " MAE on y>0 rows:", mae(m.loc[y>0,'pred6'], y[y>0]))


# ---- cell ----

import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
p6 = agent_api.load_saved('pred_e006.parquet')
print(feats.dtypes.head(3))
print(t.dtypes)
print(p6.dtypes)
print(feats['household_key'].head(), t['household_key'].head(), p6['household_key'].head())
print("pred6 cols:", p6.columns.tolist())
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
print("train rows after merge:", tr.shape)


# ---- cell ----

import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
p4 = agent_api.load_saved('pred_e004.parquet')
p5 = agent_api.load_saved('pred_e005.parquet')
p6 = agent_api.load_saved('pred_e006.parquet')

tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
base = tr[['household_key','snapshot_day','future_spend_4w','spend_28','spend_56','spend_84']]
print("base:", base.shape)
m = base.merge(p6.rename(columns={'prediction':'pred6'}), on=['household_key','snapshot_day'], how='inner')
print("after p6:", m.shape)
m = m.merge(p4.rename(columns={'prediction':'pred4'}), on=['household_key','snapshot_day'], how='inner')
print("after p4:", m.shape)
m = m.merge(p5.rename(columns={'prediction':'pred5'}), on=['household_key','snapshot_day'], how='inner')
print("after p5:", m.shape)
print(m.head())


# ---- cell ----

import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
p6 = agent_api.load_saved('pred_e006.parquet')

tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
print("tr keys sample:", tr[['household_key','snapshot_day']].head())
print("p6 keys sample:", p6[['household_key','snapshot_day']].head())
print("tr snapshot days:", sorted(tr['snapshot_day'].unique()))
print("p6 snapshot days:", sorted(p6['snapshot_day'].unique()))
print("tr hh max:", tr['household_key'].max(), "p6 hh max:", p6['household_key'].max())
k1 = set(zip(tr['household_key'], tr['snapshot_day']))
k2 = set(zip(p6['household_key'], p6['snapshot_day']))
print("key overlap:", len(k1 & k2), "tr keys:", len(k1), "p6 keys:", len(k2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')

FCOLS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]
hold  = tr[tr.snapshot_day == 431]
print("fit rows:", len(trfit), "hold rows:", len(hold))

Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh   = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def xgb_fit(X, y, seed=0, obj='reg:squarederror', lr=0.02, n=2000, md=7, mcw=10):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                         subsample=0.8, colsample_bytree=0.8, objective=obj,
                         random_state=seed, n_jobs=8, tree_method='hist')
    m.fit(X, y); return m

m1 = xgb_fit(Xtr, ytr, seed=1)
m2 = xgb_fit(Xtr, ytr, seed=2, obj='reg:quantileerror', quantile_alpha=0.5)
p1, p2 = m1.predict(Xh), m2.predict(Xh)
blend = 0.5*p1 + 0.5*p2
def mae(a,b): return np.mean(np.abs(np.asarray(a)-np.asarray(b)))
print("holdout(431) MAE sq:", round(mae(p1,yh),3), " med:", round(mae(p2,yh),3), " blend:", round(mae(blend,yh),3))
print("mean y:", round(yh.mean(),2), "mean blend:", round(blend.mean(),2),
      "median y:", round(np.median(yh),2), "median blend:", round(np.median(blend),2))
print("corr sq/med preds:", np.corrcoef(p1,p2)[0,1])


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')

FCOLS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]
hold  = tr[tr.snapshot_day == 431]
Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh   = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def xgb_fit(X, y, seed=0, obj='reg:squarederror', lr=0.02, n=2000, md=7, mcw=10, q=None):
    kw = dict(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
              subsample=0.8, colsample_bytree=0.8, objective=obj,
              random_state=seed, n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw)
    m.fit(X, y); return m

m1 = xgb_fit(Xtr, ytr, seed=1)
m2 = xgb_fit(Xtr, ytr, seed=2, obj='reg:quantileerror', q=0.5)
p1, p2 = m1.predict(Xh), m2.predict(Xh)
blend = 0.5*p1 + 0.5*p2
def mae(a,b): return np.mean(np.abs(np.asarray(a)-np.asarray(b)))
print("holdout(431) MAE sq:", round(mae(p1,yh),3), " med:", round(mae(p2,yh),3), " blend:", round(mae(blend,yh),3))
print("mean y:", round(yh.mean(),2), "mean blend:", round(blend.mean(),2),
      "median y:", round(np.median(yh),2), "median blend:", round(np.median(blend),2))
print("corr sq/med preds:", round(np.corrcoef(p1,p2)[0,1],4))
# bias structure
for nm,p in [('sq',p1),('med',p2),('blend',blend)]:
    print(nm, "MAE y==0 rows:", round(mae(p[yh==0], 0),2), " MAE y>0 rows:", round(mae(p[yh>0], yh[yh>0]),2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

# Diagnostic at snapshot 431 (allowed view) + train targets
v = agent_api.snapshot(as_of_day=431)
tx = v.table('transactions')
print("tx max day:", tx['day'].max(), "n:", len(tx))
t = agent_api.train_targets()
y431 = t[t.snapshot_day==431].set_index('household_key')['future_spend_4w']
hh = y431.index

def win_spend(tx, hh, lo, hi):
    d = tx[(tx.day>lo)&(tx.day<=hi)&(tx.household_key.isin(hh))]
    return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)

s28  = win_spend(tx, hh, 431-28, 431)
s56  = win_spend(tx, hh, 431-56, 431)
s84  = win_spend(tx, hh, 431-84, 431)
s364 = win_spend(tx, hh, 431-364, 431-336)   # same 4-week window one year ago
s336 = win_spend(tx, hh, 431-392, 431-364)   # window 1yr+4wk ago
y = y431.values
def corr(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float); m=~np.isnan(a)&~np.isnan(b)
    return np.corrcoef(a[m],b[m])[0,1]
print("corr y~s28:", round(corr(s28,y),3), " y~s56:", round(corr(s56,y),3),
      " y~s84:", round(corr(s84,y),3), " y~s364:", round(corr(s364,y),3), " y~s336:", round(corr(s336,y),3))

# partial: does s364 add beyond s28+s84? quick 2-feature OOF-ish check via binning
df = pd.DataFrame({'y':y,'s28':s28.values,'s84':s84.values,'s364':s364.values,'s336':s336.values})
df['s28b'] = pd.qcut(df.s28.rank(method='first'), 5, labels=False, duplicates='drop')
df['s364b'] = pd.qcut(df.s364.rank(method='first'), 5, labels=False, duplicates='drop')
print(df.groupby(['s28b','s364b'], observed=True)['y'].mean().unstack().round(1))

# zero structure
print("\nP(y==0 | s28==0):", round((df.loc[df.s28==0,'y']==0).mean(),3), " n:", (df.s28==0).sum())
print("P(y==0 | s28>0):", round((df.loc[df.s28>0,'y']==0).mean(),3))
print("mean y | s28==0:", round(df.loc[df.s28==0,'y'].mean(),2), " mean y | s28>0:", round(df.loc[df.s28>0,'y'].mean(),2))
print("mean s364 | s28==0:", round(df.loc[df.s28==0,'s364'].mean(),2))
print("mean y | s28==0 & s364>50:", round(df.loc[(df.s28==0)&(df.s364>50),'y'].mean(),2), " n:", ((df.s28==0)&(df.s364>50)).sum())


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.train_targets()
v = agent_api.snapshot(as_of_day=431)
tx = v.table('transactions')
y431 = t[t.snapshot_day==431].set_index('household_key')['future_spend_4w']
hh = y431.index

def win_spend(tx, hh, lo, hi):
    d = tx[(tx.day>lo)&(tx.day<=hi)&(tx.household_key.isin(hh))]
    return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)

s28 = win_spend(tx, hh, 403, 431)
y = y431.values

# --- Hypothesis A: per-household multiplicative drift, pooled shrinkage ---
df = pd.DataFrame({'y':y,'s28':s28.values})
df['hh'] = hh.values
# historical realized ratio: future4w / prior28d, using earlier train snapshots? We only have y at train snapshots.
# Instead estimate from data directly: for each household, ratio of (431->459 spend) is unknown at 431.
# Use "next 4w / last 4w" computed on PAST days (pseudo-targets): for day d in 336..403, y_pseudo(d)=spend(d+1..d+28), x=spend(d-27..d)
# do it for a few d to estimate per-household ratio with shrinkage
pseudo = []
for d in [347, 375, 403]:
    yp = win_spend(tx, hh, d+1, d+28)
    xp = win_spend(tx, hh, d-27, d)
    pseudo.append(pd.DataFrame({'hh':hh.values,'d':d,'yp':yp.values,'xp':xp.values}))
ps = pd.concat(pseudo)
ps['ratio'] = ps.yp/ps.xp.clip(lower=1e-6)
# per-household median ratio over pseudo windows (only where xp>0)
g = ps[ps.xp>10].groupby('hh')['ratio'].median()
print("pseudo-ratio distribution:", g.describe().round(3).to_dict())
print("share ratio>1.5:", (g>1.5).mean().round(3), " share ratio<0.67:", (g<0.667).mean().round(3))

# how predictive is per-household ratio (shrunken) for y at 431?
df = df.merge(g.rename('hratio'), left_on='hh', right_index=True, how='left')
df['hratio'] = df['hratio'].fillna(1.0)
# shrink toward 1
for lam in [0, 1, 3, 10]:
    r = (df.hratio*g.size + lam*1.0)/(g.size+lam)
    pred = df.s28 * ((df.hratio*g.size + lam*1.0)/(g.size+lam))
    print(f"lam={lam}: MAE={np.mean(np.abs(pred-y)):.2f}  corr={np.corrcoef(pred,y)[0,1]:.3f}")
print("baseline MAE s28:", np.mean(np.abs(df.s28-y)))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
FCOLS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]; hold = tr[tr.snapshot_day == 431]
Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def fit(Xa, ya, seed=1, obj='reg:squarederror', q=None):
    kw = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.8, objective=obj, random_state=seed,
              n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw); m.fit(Xa, ya); return m

def ev(Xa, ya, Xb, yb, label):
    m1 = fit(Xa, ya, seed=1); m2 = fit(Xa, ya, seed=2, obj='reg:quantileerror', q=0.5)
    p = 0.5*m1.predict(Xb)+0.5*m2.predict(Xb)
    print(label, "MAE:", round(np.mean(np.abs(p-yb)),3))
    return p

p_base = ev(Xtr, ytr, Xh, yh, "base(92f)")

# ---- new candidate features at snapshot 431 ----
v = agent_api.snapshot(431); tx = v.table('transactions')
hh = hold['household_key'].values
hh_idx = pd.Index(hh)
def wsum(lo, hi):
    d = tx[(tx.day>lo)&(tx.day<=hi)]
    return d.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0).values
s336, s448 = wsum(95,431), wsum(0,431)
s364, s392 = wsum(39,67), wsum(11,39)
# weeks active in last 84d
d84 = tx[(tx.day>347)&(tx.day<=431)]
wk = d84.assign(wk=(d84.day+8)//7).groupby('household_key')['wk'].nunique().reindex(hh_idx).fillna(0).values
# max single-week spend last 84d
ws = d84.assign(wk=(d84.day+8)//7).groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').max().reindex(hh_idx).fillna(0).values
# weekly spend volatility last 112d
d112 = tx[(tx.day>319)&(tx.day<=431)]
wv = d112.assign(wk=(d112.day+8)//7).groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').std().reindex(hh_idx).fillna(0).values
# exp decay half-life 56
w = 0.5**((431-tx.day)/56.0)
ed = (tx.sales_value*w).groupby(tx.household_key).sum().reindex(hh_idx).fillna(0).values

NEW = np.column_stack([s336, s448, s364, s392, wk, ws, wv, ed])
Xh2 = np.column_stack([Xh, NEW])
print("new feats corr with y:", [round(np.corrcoef(NEW[:,i], yh)[0,1],3) for i in range(NEW.shape[1])])
ev(np.column_stack([Xtr, np.tile(0,(len(Xtr),8))]), ytr, Xh2, yh, "base+8new(train-na)")  # will fail; skip proper


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

def fn(view, sd):
    tx = view.table('transactions')
    hh = pd.Index(view.households, name='household_key')
    def wsum(lo, hi=None):
        d = tx[tx.day > lo] if hi is None else tx[(tx.day > lo) & (tx.day <= hi)]
        return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    f = pd.DataFrame(index=hh)
    f['spend_336'] = wsum(sd - 336)
    f['spend_yago_4w'] = wsum(sd - 392, sd - 364)
    d84 = tx[tx.day > sd - 84].copy(); d84['wk'] = (d84.day + 8) // 7
    f['wk_active_84'] = d84.groupby('household_key')['wk'].nunique().reindex(hh).fillna(0)
    f['maxwk_84'] = d84.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').max().reindex(hh).fillna(0)
    d112 = tx[tx.day > sd - 112].copy(); d112['wk'] = (d112.day + 8) // 7
    f['wkstd_112'] = d112.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').std().reindex(hh).fillna(0)
    for hl in (28, 56, 112):
        w = 0.5 ** ((sd - tx.day) / float(hl))
        f[f'expdecay_{hl}'] = (tx.sales_value * w).groupby(tx.household_key).sum().reindex(hh).fillna(0)
    b = tx.drop_duplicates('basket_id')
    wb = 0.5 ** ((sd - b.day) / 56.0)
    f['expdecay_b56'] = wb.groupby(b.household_key).sum().reindex(hh).fillna(0)
    return f

fb = agent_api.build_features(fn)
print("build_features:", fb.shape)
feats = agent_api.load_saved('feats_v3.parquet')
full = feats.merge(fb.drop(columns=['household_key','snapshot_day'], errors='ignore') if 'snapshot_day' in fb.columns else fb,
                   left_on=['household_key','snapshot_day'], right_index=True, how='inner')
print("full:", full.shape)
agent_api.save_table(full, 'feats_v4.parquet')


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

def fn(view, sd):
    tx = view.table('transactions')
    hh = pd.Index(view.households, name='household_key')
    def wsum(lo, hi=None):
        d = tx[tx.day > lo] if hi is None else tx[(tx.day > lo) & (tx.day <= hi)]
        return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    f = pd.DataFrame(index=hh)
    f['spend_336'] = wsum(sd - 336)
    f['spend_yago_4w'] = wsum(sd - 392, sd - 364)
    d84 = tx[tx.day > sd - 84].copy(); d84['wk'] = (d84.day + 8) // 7
    f['wk_active_84'] = d84.groupby('household_key')['wk'].nunique().reindex(hh).fillna(0)
    f['maxwk_84'] = d84.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').max().reindex(hh).fillna(0)
    d112 = tx[tx.day > sd - 112].copy(); d112['wk'] = (d112.day + 8) // 7
    f['wkstd_112'] = d112.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').std().reindex(hh).fillna(0)
    for hl in (28, 56, 112):
        w = 0.5 ** ((sd - tx.day) / float(hl))
        f[f'expdecay_{hl}'] = (tx.sales_value * w).groupby(tx.household_key).sum().reindex(hh).fillna(0)
    b = tx.drop_duplicates('basket_id')
    wb = 0.5 ** ((sd - b.day) / 56.0)
    f['expdecay_b56'] = wb.groupby(b.household_key).sum().reindex(hh).fillna(0)
    return f

fb = agent_api.build_features(fn)
fb = fb.reset_index()
print("build_features:", fb.shape)
feats = agent_api.load_saved('feats_v3.parquet')
full = feats.merge(fb, on=['household_key','snapshot_day'], how='inner')
print("full:", full.shape)
agent_api.save_table(full, 'feats_v4.parquet')


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

full = agent_api.load_saved('feats_v4.parquet')
t = agent_api.train_targets()
tr = full.merge(t, on=['household_key','snapshot_day'], how='inner')
FCOLS = [c for c in full.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]; hold = tr[tr.snapshot_day == 431]
Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def fit(Xa, ya, seed=1, obj='reg:squarederror', q=None):
    kw = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.8, objective=obj, random_state=seed,
              n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw); m.fit(Xa, ya); return m

m1 = fit(Xtr, ytr, seed=1); m2 = fit(Xtr, ytr, seed=2, obj='reg:quantileerror', q=0.5)
p = 0.5*m1.predict(Xh) + 0.5*m2.predict(Xh)
def mae(a,b): return np.mean(np.abs(np.asarray(a)-np.asarray(b)))
print("holdout(431) blend MAE with 10 new features:", round(mae(p, yh),3), " (base was 64.549)")
print("mean pred:", round(p.mean(),2), "median pred:", round(np.median(p),2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

full = agent_api.load_saved('feats_v4.parquet')
t = agent_api.train_targets()
tr = full.merge(t, on=['household_key','snapshot_day'], how='inner')
FCOLS = [c for c in full.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[FCOLS].values.astype(float); ytr = tr['future_spend_4w'].values
val = full[full.snapshot_day >= 459]
Xv = val[FCOLS].values.astype(float)
print("train:", Xtr.shape, "val:", Xv.shape)

def fit(seed=1, obj='reg:squarederror', q=None):
    kw = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.8, objective=obj, random_state=seed,
              n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw); m.fit(Xtr, ytr); return m

m1 = fit(1); m2 = fit(2, obj='reg:quantileerror', q=0.5)
pred = 0.5*m1.predict(Xv) + 0.5*m2.predict(Xv)
out = val[['household_key','snapshot_day']].copy()
out['prediction'] = pred.astype(np.float32)
print(out.shape, out['prediction'].describe().round(2).to_dict())
agent_api.save_table(out, 'pred_e007.parquet')
