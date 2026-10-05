import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
print("T shape", T.shape)
print("snapshot days", sorted(T.snapshot_day.unique()))
print(T.dtypes.value_counts())
obj_cols = [c for c in T.columns if T[c].dtype == object]
print("object cols:", obj_cols)
print("NaN frac (top10):")
print(T.isna().mean().sort_values(ascending=False).head(10))
print("tt shape", tt.shape)
print(tt.future_spend_4w.describe())
print("tt days", sorted(tt.snapshot_day.unique()))
print("n feature cols:", T.shape[1]-2)
print(list(T.columns)[:60])


# ---- cell ----
import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
cols = [c for c in T.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))


# ---- cell ----
import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged", df.shape)

y = df.future_spend_4w.values
sp28 = df.spend_28.values
sp84 = df.spend_84.values
sp182 = df.spend_182.values

# overall correlations
def corr(a,b):
    m = np.isfinite(a)&np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0,1]
for c in ['spend_28','spend_84','spend_182','spend_365','d_ewma_spend_hl28','d_ewma_spend_hl112','blk_1','spend_s364','total_all','avg28_all','tenure','recency']:
    print(f"corr(y, {c}) = {corr(df[c].values.astype(float), y):.3f}")

# ratio y / trailing spend by snapshot day
for day in sorted(df.snapshot_day.unique()):
    sub = df[df.snapshot_day==day]
    m = sub.spend_28.values > 0
    r = sub.future_spend_4w.values[m] / sub.spend_28.values[m]
    print(f"day {day}: n={len(sub)}, median ratio y/sp28={np.median(r):.2f}, mean sp28={sub.spend_28.mean():.0f}, mean y={sub.future_spend_4w.mean():.0f}")

# per-household: does trailing 28d spend persist across snapshots?
T2 = T[['household_key','snapshot_day','spend_28','spend_84','spend_182','spend_365','tenure','recency']].sort_values(['household_key','snapshot_day'])
g = T2.groupby('household_key')
for c in ['spend_28','spend_84','spend_182','spend_365']:
    T2[c+'_prev'] = g[c].shift(1)
m = T2.spend_28_prev.notna() & (T2.spend_28_prev>0)
print("corr(prev spend_28, spend_28) =", np.corrcoef(T2.spend_28_prev[m], T2.spend_28[m])[0,1])
m84 = T2.spend_84_prev.notna() & (T2.spend_84_prev>0)
print("corr(prev spend_84, spend_84) =", np.corrcoef(T2.spend_84_prev[m84], T2.spend_84[m84])[0,1])
m182 = T2.spend_182_prev.notna()
print("corr(prev spend_182, spend_182) =", np.corrcoef(T2.spend_182_prev[m182], T2.spend_182[m182])[0,1])


# ---- cell ----
import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
def mae(p): return float(np.mean(np.abs(np.asarray(p)-y)))

blk = df[[f'blk_{i}' for i in range(1,14)]].values.astype(float)
med_blk = np.nanmedian(blk, axis=1)
trim_blk = np.where(np.isnan(blk), 0, blk)
top = np.nanmax(np.where(np.isnan(blk), -1, blk), axis=1)
trim_sum = np.nansum(np.where(blk==top[:,None], 0, np.where(np.isnan(blk),0,blk)), axis=1)
nobs = np.sum(~np.isnan(blk), axis=1)
trim_blk_mean = np.where(nobs>1, trim_sum/np.maximum(nobs-1,1), med_blk)

cands = {
 'global_mean': np.full(len(y), y.mean()),
 'spend_28': df.spend_28.values,
 '0.9*spend_28': 0.9*df.spend_28.values,
 'spend_84': df.spend_84.values,
 'avg28_all': df.avg28_all.values,
 'ewma_hl28': df.d_ewma_spend_hl28.values,
 'med_blk': med_blk,
 'trim_blk': trim_blk_mean,
 'blend .5s84+.3s28+.2avg': 0.5*df.spend_84.values+0.3*df.spend_28.values+0.2*df.avg28_all.values,
 'blend .4s84+.3s28+.3avg': 0.4*df.spend_84.values+0.3*df.spend_28.values+0.3*df.avg28_all.values,
 'max(s28, .8*s84)': np.maximum(df.spend_28.values, 0.8*df.spend_84.values),
}
for k,v in cands.items(): print(f"{k:28s} MAE {mae(v):8.2f}")

# ridge on a few top features, day-holdout (fit <=403, predict 431)
feats = ['spend_28','spend_84','spend_182','spend_365','avg28_all','d_ewma_spend_hl28','d_ewma_spend_hl112',
         'trips_28','trips_84','recency','tenure','blk_1','d_wk_spend_mean_12w','basket_mean_84','zero6','decay_mean']
X = df[feats].fillna(0).values.astype(float)
def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha  # no penalty on intercept col
    Z1 = np.hstack([Z, np.ones((len(Z),1))])
    w = np.linalg.solve(A, Z1.T@ytr)
    return mu, sd, w
def ridge_pred(Xte, m):
    mu, sd, w = m
    Z = (Xte-mu)/sd
    return np.hstack([Z, np.ones((len(Z),1))])@w
d = df.snapshot_day.values.astype(int)
tr, te = d<=403, d==431
best=None
for alpha in [1,10,100,1000]:
    m = ridge_fit(X[tr], y[tr], alpha)
    p = ridge_pred(X[te], m)
    mm = mae(p[te]) if False else float(np.mean(np.abs(p-y[te])))
    print(f"ridge alpha={alpha}: holdout(431) MAE {mm:.2f}")
m = ridge_fit(X, y, 100)
print("ridge alpha=100 in-sample MAE:", float(np.mean(np.abs(ridge_pred(X,m)-y))))

# probe: load_saved inside build_features
def probe(view, snapshot_day):
    e = agent_api.load_saved('e012_style.parquet')
    e = e[e.snapshot_day==snapshot_day]
    out = e.set_index('household_key').drop(columns=['snapshot_day'])
    out['probe_ok'] = 1.0
    return out
res = agent_api.build_features(probe)
print("probe result:", res.shape, "probe_ok sum:", res.probe_ok.sum(), "days:", sorted(res.snapshot_day.unique()))


# ---- cell ----
import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
def mae(p): return float(np.mean(np.abs(np.asarray(p)-y)))

# scale-adjusted candidates
cands = {
 'sp28': df.spend_28.values,
 '0.9*sp28': 0.9*df.spend_28.values,
 's84/3': df.spend_84.values/3,
 's182/6.5': df.spend_182.values/6.5,
 's365/13': df.spend_365.values/13,
 'avg28_all': df.avg28_all.values,
 'blend .5(s84/3)+.3s28+.2avg': 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values,
 'blend .4(s84/3)+.3s28+.3avg': 0.4*df.spend_84.values/3+0.3*df.spend_28.values+0.3*df.avg28_all.values,
 'max(s28, s84/3)': np.maximum(df.spend_28.values, df.spend_84.values/3),
 'ewma_hl28': df.d_ewma_spend_hl28.values,
 '0.5*ewma28+0.5*avg28': 0.5*df.d_ewma_spend_hl28.values+0.5*df.avg28_all.values,
}
for k,v in cands.items(): print(f"{k:30s} MAE {mae(v):8.2f}")

feats = ['spend_28','spend_84','spend_182','spend_365','avg28_all','d_ewma_spend_hl28','d_ewma_spend_hl112',
         'trips_28','trips_84','recency','tenure','blk_1','d_wk_spend_mean_12w','basket_mean_84','zero6','decay_mean']
X = df[feats].fillna(0).values.astype(float)
X[:,1] = X[:,1]/3.0; X[:,2]=X[:,2]/6.5; X[:,3]=X[:,3]/13.0  # scale-adjust
X = np.hstack([X, np.ones((len(X),1))])
def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr[:,:-1].mean(0), Xtr[:,:-1].std(0); sd[sd==0]=1
    Z = (Xtr[:,:-1]-mu)/sd
    Z1 = np.hstack([Z, np.ones((len(Z),1))])
    A = Z1.T@Z1 + alpha*np.eye(Z1.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z1.T@ytr)
    return mu, sd, w
def ridge_pred(Xte, m):
    mu, sd, w = m
    Z = (Xte[:,:-1]-mu)/sd
    return np.hstack([Z, np.ones((len(Z),1))])@w
d = df.snapshot_day.values.astype(int)
tr, te = d<=403, d==431
for alpha in [1,10,100,1000]:
    m = ridge_fit(X[tr], y[tr], alpha)
    print(f"ridge alpha={alpha}: holdout(431) MAE {mae(ridge_pred(X[te],m)):.2f}")
m = ridge_fit(X, y, 100)
print("ridge alpha=100 in-sample MAE:", mae(ridge_pred(X,m)))
# day-holdout for simple candidates too
for k in ['sp28','0.9*sp28','avg28_all','blend .5(s84/3)+.3s28+.2avg']:
    v = cands[k]
    print(f"{k:30s} holdout(431) MAE {mae(v[te]):.2f}")


# ---- cell ----
import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
print("blend MAE", np.mean(np.abs(resid)).round(2))

# MAE by y decile
q = pd.qcut(y, 10, duplicates='drop')
print(pd.DataFrame({'y':y,'absres':np.abs(resid)}).groupby(q, observed=True).agg(n=('y','size'), mae=('absres','mean'), ymean=('y','mean'), pmean=('pred','mean')).round(1))

# feature correlation with residual
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
rows=[]
for c in num:
    a = df[c].astype(float).values
    m = np.isfinite(a)
    if m.sum()<1000: continue
    aa = a.copy(); aa[~m]=np.nan
    # fillna with median for corr
    med = np.nanmedian(aa)
    aa = np.where(np.isfinite(aa), aa, med)
    if aa.std()==0: continue
    r = np.corrcoef(aa, resid)[0,1]
    rows.append((abs(r), r, c))
rows.sort(reverse=True)
print("\nTop |corr| with residual (blend):")
for _,r,c in rows[:25]: print(f"  {r:+.3f} {c}")
print("\nLowest:")
for _,r,c in rows[-8:]: print(f"  {r:+.3f} {c}")


# ---- cell ----
import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
tmp = pd.DataFrame({'y':y,'absres':np.abs(resid),'pred':pred})
q = pd.qcut(y, 10, duplicates='drop')
print(tmp.groupby(q, observed=True).agg(n=('y','size'), mae=('absres','mean'), ymean=('y','mean'), pmean=('pred','mean')).round(1))

num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
rows=[]
for c in num:
    a = df[c].astype(float).values
    m = np.isfinite(a)
    if m.sum()<1000: continue
    med = np.nanmedian(a)
    aa = np.where(m, a, med)
    if aa.std()==0: continue
    r = np.corrcoef(aa, resid)[0,1]
    rows.append((abs(r), r, c))
rows.sort(reverse=True)
print("\nTop |corr| with residual (blend):")
for _,r,c in rows[:25]: print(f"  {r:+.3f} {c}")


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.values.astype(int)

num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
X = df[num].astype(float).copy()
# fill NaN with train medians
tr_all = d<=403
med = X[tr_all].median()
X = X.fillna(med).values.astype(float)
# scale-adjust the window sums
for c in ['spend_84','spend_112']: X[:, num.index(c)]/=3.0
for c in ['spend_182']: X[:, num.index(c)]/=6.5
for c in ['spend_365','total_all']: X[:, num.index(c)]/=13.0
for c in ['blk_1','blk_2','blk_3','blk_4','blk_5','blk_6','blk_7','blk_8','blk_9','blk_10','blk_11','blk_12','blk_13','spend_s336','spend_s364','spend_s392','max6','d_block_max']:
    X[:, num.index(c)] = X[:, num.index(c)]  # blocks are 28d sums, fine

X1 = np.hstack([X, np.ones((len(X),1))])
def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr[:,:-1].mean(0), Xtr[:,:-1].std(0); sd[sd==0]=1
    Z = np.hstack([(Xtr[:,:-1]-mu)/sd, np.ones((len(Xtr),1))])
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    return (mu, sd, np.linalg.solve(A, Z.T@ytr))
def ridge_pred(Xte, m):
    mu, sd, w = m
    return np.hstack([(Xte[:,:-1]-mu)/sd, np.ones((len(Xte),1))])@w

tr, te = d<=403, d==431
for alpha in [3,30,300,3000]:
    m = ridge_fit(X1[tr], y[tr], alpha)
    print(f"ridge-all alpha={alpha}: holdout(431) MAE {np.mean(np.abs(ridge_pred(X1[te],m)-y[te])):.2f}")

# heavy tail: within top decile, what correlates with residual?
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
heavy = y > 366
rows=[]
for c in num:
    a = df[c].astype(float).values
    m = np.isfinite(a) & heavy
    if m.sum()<500: continue
    med2 = np.nanmedian(a[heavy])
    aa = np.where(np.isfinite(a), a, med2)
    if aa[heavy].std()==0: continue
    r = np.corrcoef(aa[heavy], resid[heavy])[0,1]
    rows.append((abs(r), r, c))
rows.sort(reverse=True)
print("\nHeavy (y>366, n=%d) residual corr:" % heavy.sum())
for _,r,c in rows[:20]: print(f"  {r:+.3f} {c}")


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.values.astype(int)
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
X = df[num].astype(float).copy()
print("inf counts:", np.isinf(X.values).sum())
X = X.replace([np.inf,-np.inf], np.nan)
tr_all = d<=403
med = X[tr_all].median()
X = X.fillna(med).values.astype(float)
for c,i in [('spend_84',num.index('spend_84')),('spend_112',num.index('spend_112'))]: X[:,i]/=3.0
X[:,num.index('spend_182')]/=6.5
for c in ['spend_365','total_all']: X[:,num.index(c)]/=13.0

def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    Z = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))])
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    return (mu, sd, np.linalg.solve(A, Z.T@ytr))
def ridge_pred(Xte, m):
    mu, sd, w = m
    return np.hstack([(Xte-mu)/sd, np.ones((len(Xte),1))])@w
tr, te = d<=403, d==431
for alpha in [3,30,300,3000]:
    m = ridge_fit(X[tr], y[tr], alpha)
    print(f"ridge-all alpha={alpha}: holdout(431) MAE {np.mean(np.abs(ridge_pred(X[te],m)-y[te])):.2f}")
m = ridge_fit(X[tr_all], y[tr_all], 30)
pv = ridge_pred(X[d>=459], m)
print("val preds: mean %.1f, neg %d" % (pv.mean(), (pv<0).sum()))

# heavy segment: regress residual on features (ridge, small alpha)
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
heavy = (y>366) & tr_all
Xh, rh = X[heavy], resid[heavy]
mh = ridge_fit(Xh, rh, 300)
ph = ridge_pred(Xh, mh)
print("\nheavy resid ridge in-sample R2:", 1 - np.var(rh-ph)/np.var(rh))
# top positive weights
w = mh[2][:-1]
order = np.argsort(-w)[:15]
print("top positive weights (heavy resid):")
for i in order: print(f"  {w[i]:+.2f} {num[i]}")


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
X = df[num].astype(float).copy()
X = X.replace([np.inf,-np.inf], np.nan)
print("NaN per col max:", X.isna().sum().max())
print("cols all-NaN:", [c for c in num if X[c].isna().all()])
X = X.fillna(X.median())
print("any NaN left:", np.isnan(X.values).any(), "any inf:", np.isinf(X.values).any())
print("y NaN:", np.isnan(y).sum())
print("y max:", y.max())
A = X.T@X
print("A any nan:", np.isnan(A).any(), "A max:", np.nanmax(A))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

def _idx(h):
    if isinstance(h, pd.DataFrame): return h.index
    return pd.Index(h)

# ---- probe: can fn() use load_saved? ----
def probe(view, sd):
    ok = 0.0
    try:
        e = agent_api.load_saved('e012_style.parquet')
        e = e[e.snapshot_day == sd]
        ok = 2.0 if len(e) > 0 else 0.5
    except Exception:
        ok = 0.0
    return pd.DataFrame({'ok': [ok]}, index=_idx(view.households))

res = agent_api.build_features(probe)
vals = sorted(res.ok.value_counts().to_dict().items())
print("PROBE:", vals)
LOAD_OK = (len(vals) == 1 and vals[0][0] == 2.0)
print("LOAD_OK:", LOAD_OK)

def fn(view, sd):
    idx = _idx(view.households)
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(set(idx))].copy()
    t = float(sd)
    tx['age'] = t - tx.day.values.astype(float)
    hh = tx.household_key.values
    sv = tx.sales_value.values.astype(float)
    age = tx.age.values
    out = pd.DataFrame(index=idx)

    def agg(mask, val):
        s = pd.Series(np.asarray(val, dtype=float)[mask], index=hh[mask])
        return s.groupby(level=0).sum().reindex(idx).fillna(0.0)

    for k in range(1, 9):  # weekly spend buckets, most recent first
        out[f'w{k}'] = agg((age < 7*k) & (age >= 7*(k-1)), sv)
    out['spend_168'] = agg(age < 168, sv)
    for hl in [7.0, 21.0, 42.0, 84.0]:  # finer EWMA grid
        out[f'e_hl{int(hl)}'] = agg(age <= 365, sv * 0.5**(age/hl))
    # EWMA trips (per basket), hl 28
    b = tx[age <= 365].groupby(['household_key','basket_id']).agg(bday=('day','first'))
    bd = t - b.bday.values.astype(float)
    out['e_trips_hl28'] = pd.Series(0.5**(bd/28.0), index=b.index.get_level_values(0))\
        .groupby(level=0).sum().reindex(idx).fillna(0.0)
    # weekly series stats (12 weeks)
    wk = (age // 7).astype(int)
    m12 = wk < 12
    dfw = pd.DataFrame({'hh': hh[m12], 'wk': wk[m12], 'v': sv[m12]})
    W = dfw.pivot_table(index='hh', columns='wk', values='v', aggfunc='sum')\
        .reindex(columns=range(12)).reindex(idx).fillna(0.0).values
    mn = W.mean(1); s_ = W.std(1)
    out['wk_mean12'] = mn
    out['wk_cv12'] = np.where(mn > 1e-9, s_/np.maximum(mn, 1e-9), 0.0)
    a, b2 = W[:, :-1], W[:, 1:]
    am = a - a.mean(1, keepdims=True); bm = b2 - b2.mean(1, keepdims=True)
    den = np.sqrt((am**2).sum(1)*(bm**2).sum(1))
    out['wk_ac1'] = np.where(den > 1e-9, (am*bm).sum(1)/np.maximum(den, 1e-9), 0.0)
    # big-trip concentration (84d)
    m84 = age < 84
    bb = tx[m84].groupby(['household_key','basket_id']).sales_value.sum()
    g = bb.groupby(level=0)
    mx = g.max().reindex(idx).fillna(0.0)
    tot = g.sum().reindex(idx).fillna(0.0)
    out['bt_max_84'] = mx
    out['bt_cnt100_84'] = g.apply(lambda s: float((s > 100).sum())).reindex(idx).fillna(0.0)
    out['bt_top1_share_84'] = np.where(tot > 0, mx/np.where(tot > 0, tot, 1.0), np.nan)

    if LOAD_OK:
        base = agent_api.load_saved('e012_style.parquet')
        base = base[base.snapshot_day == sd].set_index('household_key').drop(columns=['snapshot_day'])
        return out.join(base, how='left')
    return out

tab = agent_api.build_features(fn)
print("built:", tab.shape)
print("has e012 cols:", 'spend_84' in tab.columns, "| has new:", 'e_hl84' in tab.columns, 'wk_ac1' in tab.columns)
p = agent_api.save_table(tab, 'e014_recency.parquet')
print("saved:", p)


# ---- cell ----
import pandas as pd
def probe(view, sd):
    msg = ''
    try:
        e = agent_api.load_saved('e012_style.parquet')
        msg = 'loaded ok'
    except Exception as ex:
        msg = f"{type(ex).__name__}: {ex}"
    try:
        has_attr = hasattr(agent_api, 'load_saved')
    except Exception:
        has_attr = '?'
    return pd.DataFrame({'msg': [msg], 'has_attr': [str(has_attr)]}, index=pd.Index(view.households))
res = agent_api.build_features(probe)
print(res.msg.value_counts().to_dict(), res.has_attr.value_counts().to_dict())
