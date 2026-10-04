print("snapshot_days:", snapshot_days())
s = snapshot()
print("txn rows<=459:", s.transactions.shape)

base = load_saved('e005_full_plus_mix.parquet')
print("base shape:", base.shape)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print("n base feature cols:", len(bcols))
print("base cols:", bcols)

mix = load_saved('mix_v1.parquet')
print("mix cols:", [c for c in mix.columns if c not in ('household_key','snapshot_day')])

tt = train_targets()
print("targets shape:", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']))

m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.astype(float)
rows=[]
for c in bcols:
    x = m[c]
    if x.dtype.kind not in 'ifb': continue
    xm = x.fillna(x.median()).astype(float)
    if xm.std()==0: continue
    rows.append((c, float(np.corrcoef(xm, y)[0,1])))
cs = pd.DataFrame(rows, columns=['f','r'])
cs['a']=cs.r.abs()
cs=cs.sort_values('a',ascending=False)
print("TOP CORR:")
print(cs.head(35).to_string())
print("BOTTOM CORR:")
print(cs.tail(8).to_string())

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def mae(p): return float(np.mean(np.abs(y-p)))
print("median pred:", mae(np.full(len(y), np.median(y))))
print("mean pred:", mae(np.full(len(y), y.mean())))
for c in ['spend_84','spend_28','spend_56','x_exp4w','spend_364']:
    x = m[c].fillna(m[c].median()).values.astype(float)
    print(c, "identity MAE:", round(mae(x),2), end='  ')
    A = np.vstack([x, np.ones(len(x))]).T
    coef,*_ = np.linalg.lstsq(A, y, rcond=None)
    print("linfit MAE:", round(mae(A@coef),2))

# per-snapshot-day mean target as predictor (calendar only)
daymean = m.groupby('snapshot_day').future_spend_4w.mean()
print("day-mean pred MAE:", round(mae(daymean.reindex(m.snapshot_day).values),2))

# distribution of y vs spend_84 ratio
r = y / np.maximum(m.spend_84.values.astype(float),1)
print("y/spend_84 ratio quantiles:", np.nanpercentile(r,[10,25,50,75,90]).round(2))
print("corr y with log1p(spend_84):", np.corrcoef(np.log1p(m.spend_84.values), y)[0,1])

# ---- cell ----
import numpy as np, pandas as pd

def structure_fn(view, snapshot_day):
    d = view.day
    keys = view.households.index if isinstance(view.households, pd.DataFrame) else pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.household_key.isin(keys)]
    out = pd.DataFrame(index=keys)

    t84 = tx[tx.day > d-84]
    g = t84.groupby('household_key')
    spend84 = g.sales_value.sum()
    out['s_gap_mean'] = (d - g.day.max())
    pdays = t84.groupby('household_key').day.apply(lambda s: np.sort(s.values))
    def gapstats(v):
        if len(v) < 2: return (np.nan, np.nan, np.nan, np.nan)
        gp = np.diff(v)
        return (gp.mean(), gp.std(), gp.max(), np.mean(gp>14))
    gs = pdays.apply(gapstats)
    out['gap_mean'] = gs.apply(lambda t: t[0])
    out['gap_std'] = gs.apply(lambda t: t[1])
    out['gap_max'] = gs.apply(lambda t: t[2])
    out['gap_gt14'] = gs.apply(lambda t: t[3])
    gm = out['gap_mean']
    out['gap_cur_ratio'] = out['s_gap_mean'] / gm.replace(0, np.nan)

    t12 = tx[tx.day > d-84]
    wk = ((t12.day + 8)//7)
    t12 = t12.assign(_w=wk)
    wact = t12.groupby('household_key')._w.nunique()
    out['wk_active_12'] = wact / 12.0
    wsp = t12.groupby(['household_key','_w']).sales_value.sum()
    ws = wsp.groupby(level=0)
    out['wk_cv'] = ws.std() / ws.mean().replace(0, np.nan)
    wcur = (d + 8)//7
    def streak(wsset):
        s = 0; w = wcur-1
        for k in sorted(wsset, reverse=True):
            if k == w: s += 1; w -= 1
            elif k < w: break
        return s
    out['streak_w'] = wact.index.to_series().map(lambda h: streak(set(wsp.loc[h].index)) if h in wsp.index else 0)

    st = t84.groupby(['household_key','store_id']).sales_value.sum()
    out['store_share_top'] = st.groupby(level=0).max() / spend84.replace(0, np.nan)
    out['n_stores84'] = t84.groupby('household_key').store_id.nunique()

    disc = (t84.retail_disc.fillna(0) + t84.coupon_disc.fillna(0) + t84.coupon_match_disc.fillna(0)).clip(lower=0)
    denom = (t84.sales_value + disc.clip(lower=0))
    out['disc_share'] = disc.groupby(t84.household_key).sum() / denom.groupby(t84.household_key).sum().replace(0, np.nan)

    upb = t84.groupby(['household_key','basket_id']).quantity.sum().groupby(level=0).mean()
    out['units_per_basket'] = upb

    tt = t84.trans_time.fillna(0)
    ev = (tt >= 1700).groupby(t84.household_key).mean()
    out['evening_share'] = ev

    pr = t84.groupby(['household_key','product_id']).sales_value.sum()
    out['prod_top_share'] = pr.groupby(level=0).max() / spend84.replace(0, np.nan)
    bk = t84.groupby(['household_key','basket_id']).sales_value.sum()
    out['basket_top_share'] = bk.groupby(level=0).max() / spend84.replace(0, np.nan)

    t28 = tx[tx.day > d-28]
    tprior = tx[(tx.day > d-112) & (tx.day <= d-28)]
    pr28 = t28.groupby(['household_key','product_id']).sales_value.sum()
    priorset = tprior.groupby('household_key').product_id.apply(set)
    def rep_share(h):
        if h not in pr28.index or h not in priorset.index: return np.nan
        s = pr28.loc[h]; ps = priorset.loc[h]
        if len(s)==0 or s.sum()==0: return np.nan
        m = pd.Series(s.index).isin(ps).values
        return float(s.values[m].sum() / s.sum())
    out['prod_repeat_28'] = pd.Series({h: rep_share(h) for h in keys})
    allprior = tx[tx.day <= d-28].groupby('household_key').product_id.apply(set)
    def new_share(h):
        if h not in pr28.index or h not in allprior.index: return np.nan
        s = pr28.loc[h]; ps = allprior.loc[h]
        if len(s)==0 or s.sum()==0: return np.nan
        m = pd.Series(s.index).isin(ps).values
        return float(s.values[~m].sum() / s.sum())
    out['prod_new_28'] = pd.Series({h: new_share(h) for h in keys})

    out['n_prod84'] = t84.groupby('household_key').product_id.nunique() / t84.groupby('household_key').basket_id.nunique().replace(0,np.nan)
    out['spend_per_active_day'] = spend84 / t84.groupby('household_key').day.nunique().replace(0,np.nan)
    return out

feats = build_features(structure_fn)
print("built:", feats.shape)
print("cols:", list(feats.columns))
print("NaN frac:")
print(feats.isna().mean().round(2).to_string())
save_table(feats.reset_index(), 'structure_v1')
print("saved")

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
st = load_saved('structure_v1.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left').merge(
    st.drop(columns=['snapshot_day'], errors='ignore') if 'snapshot_day' in st.columns else st,
    on=['household_key'], how='left', suffixes=('','_s'))
# structure table has household_key+snapshot_day index reset; check
print(st.columns[:5], st.shape)
m = tt.merge(base, on=['household_key','snapshot_day'], how='left').merge(st, on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape)
y = m.future_spend_4w.values.astype(float)

def prep(df):
    X = df.copy()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(col.median() if col.notna().any() else 0)
    return X.values

bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
scols = [c for c in st.columns if c not in ('household_key','snapshot_day')]
Xb = prep(m[bcols]); Xs = prep(m[scols])
Xall = np.hstack([Xb, Xs])

# standardize + ridge, random split proxy (train snapshots only)
tr = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values
rng = np.random.RandomState(0)
def ridge_mae(X, mask_tr, mask_va):
    mu, sd = X[mask_tr].mean(0), X[mask_tr].std(0)+1e-9
    Z = (X-mu)/sd
    Zt = np.hstack([Z, np.ones((len(Z),1))])
    lam = 30.0
    A = Zt[mask_tr].T@Zt[mask_tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[mask_tr].T@y[mask_tr])
    return float(np.mean(np.abs(y[mask_va] - Zt[mask_va]@w)))

# proxy split within train snapshots
va = tr & (rng.rand(len(y))<0.25)
trp = tr & ~va
print("ridge base MAE:", round(ridge_mae(Xb, trp, va),3))
print("ridge base+structure MAE:", round(ridge_mae(Xall, trp, va),3))
# partial corr of each structure feature with y, controlling base ridge fit
mu, sd = Xb[trp].mean(0), Xb[trp].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[trp].T@Zt[trp] + 30*np.eye(Zt.shape[1]); A[-1,-1]-=30
w = np.linalg.solve(A, Zt[trp].T@y[trp])
resid = y - Zt@w
rows=[]
for j,c in enumerate(scols):
    x = Xs[:,j]
    if x.std()==0: continue
    rows.append((c, float(np.corrcoef(x, resid)[0,1])))
pc = pd.DataFrame(rows, columns=['f','pr']); pc['a']=pc.pr.abs()
print(pc.sort_values('a',ascending=False).head(22).to_string())

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
st = load_saved('structure_v1.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left').merge(st, on=['household_key','snapshot_day'], how='left', suffixes=('','_s'))
y = m.future_spend_4w.values.astype(float)

def prep(df):
    X = df.copy()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(col.median() if col.notna().any() else 0)
    return X.values

bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
scols = [c for c in st.columns if c not in ('household_key','snapshot_day')]
Xb = prep(m[bcols]); Xs = prep(m[scols])
Xall = np.hstack([Xb, Xs])

tr = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values
rng = np.random.RandomState(0)
def ridge_mae(X, mask_tr, mask_va, lam=30.0):
    mu, sd = X[mask_tr].mean(0), X[mask_tr].std(0)+1e-9
    Z = (X-mu)/sd
    Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[mask_tr].T@Zt[mask_tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[mask_tr].T@y[mask_tr])
    return float(np.mean(np.abs(y[mask_va] - Zt[mask_va]@w)))

va = tr & (rng.rand(len(y))<0.25)
trp = tr & ~va
print("ridge base MAE:", round(ridge_mae(Xb, trp, va),3))
print("ridge base+structure MAE:", round(ridge_mae(Xall, trp, va),3))

mu, sd = Xb[trp].mean(0), Xb[trp].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[trp].T@Zt[trp] + 30*np.eye(Zt.shape[1]); A[-1,-1]-=30
w = np.linalg.solve(A, Zt[trp].T@y[trp])
resid = y - Zt@w
rows=[]
for j,c in enumerate(scols):
    x = Xs[:,j]
    if x.std()==0: continue
    rows.append((c, float(np.corrcoef(x, resid)[0,1])))
pc = pd.DataFrame(rows, columns=['f','pr']); pc['a']=pc.pr.abs()
print(pc.sort_values('a',ascending=False).to_string())

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med

Xb, med = prep(m[bcols])

def ridge_pred(X, tr, va, lam):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Z = (X-mu)/sd
    Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[tr].T@y[tr])
    return Zt@w

def ev(X, tr, va, lam=30.0):
    p = ridge_pred(X, tr, va, lam)
    return float(np.mean(np.abs(y[va]-p[va])))

TR13 = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values
# temporal proxy: train 95-403, validate 431
tr_t = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403]).values
va_t = (m.snapshot_day==431).values
# random proxy
rng = np.random.RandomState(0)
va_r = TR13 & (rng.rand(len(y))<0.25); tr_r = TR13 & ~va_r

for lam in [3,10,30,100,300]:
    print(f"lam={lam}: temporal {ev(Xb,tr_t,va_t,lam):.2f}  random {ev(Xb,tr_r,va_r,lam):.2f}")

# feature drift across snapshot days (train rows only)
print("\nDrift of key features by snapshot day (train rows):")
for c in ['spend_84','spend_28','spend_364','spend_all','recency','tenure','x_exp4w','zero_w12','m_spend28']:
    g = m[m.TR13 if False else m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])].groupby('snapshot_day')[c].median()
    print(c, g.round(1).values)
# validation rows feature medians (no targets)
mv = base[base.snapshot_day.isin([459,487,515,543])]
print("\nval medians:", {c: round(float(mv[c].median()),1) for c in ['spend_84','spend_28','spend_364','spend_all','recency','tenure','x_exp4w','zero_w12','m_spend28']})

# ---- cell ----
import numpy as np, pandas as pd

base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med

Xb, med = prep(m[bcols])
TR13 = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values

def bin_features(X, tr, nb=64):
    Xb_ = np.zeros(X.shape, dtype=np.int8)
    for j in range(X.shape[1]):
        qs = np.quantile(X[tr,j], np.linspace(0,1,nb+1)[1:-1])
        Xb_[:,j] = np.searchsorted(qs, X[:,j])
    return Xb_

def gbm(Xb_, y, tr_idx, rounds=120, lr=0.08, depth=3, min_leaf=150, colfrac=0.7, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    pred = np.zeros(n); pred[tr_idx] = y[tr_idx].mean()
    F = Xb_.shape[1]
    for r in range(rounds):
        resid = y - pred
        feats = rng.choice(F, int(F*colfrac), replace=False)
        def build(rows):
            if len(rows) < min_leaf: return ('leaf', resid[rows].mean())
            best = None; gr = resid[rows]
            for j in feats:
                b = Xb_[rows, j]
                c = np.bincount(b, minlength=64).astype(float)
                h = np.bincount(b, weights=gr, minlength=64)
                cs = np.cumsum(c); gs = np.cumsum(h)
                rc, rg = cs[-1], gs[-1]
                if rc < 2*min_leaf: continue
                cl = cs[:-1]; gl = gs[:-1]; cr = rc-cl; grr = rg-gl
                valid = (cl>=min_leaf)&(cr>=min_leaf)
                if not valid.any(): continue
                with np.errstate(invalid='ignore'):
                    score = np.where(valid, gl*gl/np.maximum(cl,1) + grr*grr/np.maximum(cr,1), -np.inf)
                k = int(np.argmax(score))
                if best is None or score[k] > best[0]: best = (score[k], j, k)
            if best is None: return ('leaf', gr.mean())
            _, j, k = best
            mask = Xb_[rows,j] <= k
            return ('node', j, k, build(rows[mask]), build(rows[~mask]))
        tree = build(tr_idx)
        def apply(t, rows):
            if t[0]=='leaf': pred[rows] += lr*t[1]; return
            _, j, k, L, R = t
            b = Xb_[rows, j]
            apply(L, rows[b<=k]); apply(R, rows[b>k])
        apply(tree, np.arange(n))
    return pred

Xbin = bin_features(Xb, TR13)
tr_t = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403]).values)[0]
va_t = np.where((m.snapshot_day==431).values)[0]
p = gbm(Xbin, y, tr_t)
print("GBM proxy temporal (val=431) MAE:", round(float(np.mean(np.abs(y[va_t]-p[va_t]))),3))
rng = np.random.RandomState(0)
va_r = np.where(TR13 & (rng.rand(len(y))<0.25))[0]
tr_r = np.where(TR13 & ~(rng.rand(len(y))<0.25))[0]
p2 = gbm(Xbin, y, tr_r)
print("GBM proxy random MAE:", round(float(np.mean(np.abs(y[va_r]-p2[va_r]))),3))

# ---- cell ----
import numpy as np, pandas as pd
s = snapshot()
tx = s.transactions
wk = (tx.day + 8)//7
prof = tx.assign(_w=wk).groupby('_w').sales_value.sum()
prof_n = tx.assign(_w=wk).groupby('_w').basket_id.nunique()
print("weeks:", prof.index.min(), prof.index.max())
print("weekly spend (first 30):")
print(prof.head(30).round(0).to_string())
print("top 12 spend weeks:")
print(prof.sort_values(ascending=False).head(12).round(0).to_string())
print("bottom 6:", prof.sort_values().head(6).round(0).values)
print("mean weekly:", round(prof.mean(),0), "std:", round(prof.std(),0))
# day-level pattern
dd = tx.groupby('day').sales_value.sum()
print("day-of-week effect (day mod 7):")
print(dd.groupby(dd.index % 7).mean().round(0).to_string())

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med
Xb, med = prep(m[bcols])

tr = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403]).values)[0]
va = np.where((m.snapshot_day==431).values)[0]
lam=100.0
mu, sd = Xb[tr].mean(0), Xb[tr].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
w = np.linalg.solve(A, Zt[tr].T@y[tr])
p = Zt@w
err = y - p
mv = m.iloc[va].copy(); mv['pred']=p[va]; mv['err']=err[va]; mv['y']=y[va]
mv['abserr']=np.abs(err[va])
print("val MAE:", mv.abserr.mean().round(2))
# error by predicted level decile
mv['pdec'] = pd.qcut(mv.pred, 10, duplicates='drop')
print(mv.groupby('pdec').agg(n=('abserr','size'), mae=('abserr','mean'), bias=('err','mean'), y=('y','mean'), pred=('pred','mean')).round(1).to_string())
# zero-y households
z = mv[mv.y==0]
print("\nzero-y rows:", len(z), "mean pred:", z.pred.mean().round(1), "MAE contribution:", z.abserr.mean().round(1))
nz = mv[mv.y>0]
print("nonzero rows:", len(nz), "mean pred:", nz.pred.mean().round(1))
# high spenders
h = mv.nlargest(10, 'y')[['y','pred','err','spend_84','spend_28','x_exp4w','recency','zero_w12']]
print("\ntop-10 y rows:\n", h.round(1).to_string())
# error vs spend_84 decile
mv['sdec'] = pd.qcut(mv.spend_84, 10, duplicates='drop')
print("\nby spend_84 decile:")
print(mv.groupby('sdec').agg(n=('abserr','size'), mae=('abserr','mean'), bias=('err','mean'), y=('y','mean'), pred=('pred','mean')).round(1).to_string())
# what correlates with |err| or err among base features
rows=[]
for c in bcols:
    x = mv[c]
    if x.dtype.kind not in 'ifb': continue
    x = x.astype(float).replace([np.inf,-np.inf],np.nan).fillna(x.median())
    if x.std()==0: continue
    rows.append((c, float(np.corrcoef(x, mv.err)[0,1]), float(np.corrcoef(x, mv.abserr)[0,1])))
ec = pd.DataFrame(rows, columns=['f','r_err','r_abserr'])
print("\ncorr with err / |err| (top 15 by |r_err|):")
print(ec.reindex(ec.r_err.abs().sort_values(ascending=False).index).head(15).round(3).to_string())
print("\ncorr with |err| (top 15):")
print(ec.reindex(ec.r_abserr.abs().sort_values(ascending=False).index).head(15).round(3).to_string())

# ---- cell ----
import numpy as np, pandas as pd
s = snapshot()
tx = s.transactions
wk = (tx.day + 8)//7
wsp = tx.assign(_w=wk).groupby('_w').sales_value.sum()
wsp = wsp[wsp.index>=5]  # drop onboarding weeks 1-4
# detrend: ratio to centered 13-week moving average
idx = wsp.index.values
ma = wsp.rolling(13, center=True, min_periods=5).mean()
ratio = wsp/ma
print("seasonal ratio by week (5..66):")
print(ratio.round(2).to_string())

train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
val_days = [459,487,515,543]
tt = train_targets()
daymean = tt.groupby('snapshot_day').future_spend_4w.mean()

def seasonal_index(w, mode='next4'):
    # year-ago analog: mean weekly spend in weeks w-51..w-48 (next 4 weeks a year ago)
    # relative to weeks w-55..w-52 (trailing 4 weeks a year ago)
    def meanw(ws):
        vals = [wsp.get(k, np.nan) for k in ws]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else np.nan
    nxt = meanw(range(w-51, w-47))
    trl = meanw(range(w-55, w-51))
    return nxt/trl if trl and not np.isnan(trl) and trl>0 else np.nan

rows=[]
for d in train_days:
    w = (d+8)//7
    # realized: population next-4-weeks spend vs trailing-4-weeks, using full data (analysis only)
    def meanw_real(ws):
        vals = [wsp.get(k, np.nan) for k in ws]
        return np.nanmean(vals)
    realized = meanw_real(range(w+1,w+5))/meanw_real(range(w-3,w+1))
    si = seasonal_index(w)
    rows.append((d, w, round(realized,3), round(si,3) if not np.isnan(si) else np.nan, round(daymean[d],1)))
df = pd.DataFrame(rows, columns=['day','week','realized_next4/trail4','yearago_index','target_mean'])
print(df.to_string())
print("\ncorr realized vs target_mean:", np.corrcoef(df.realized[3:], df.target_mean[3:])[0,1])
print("corr yearago_index vs target_mean:", np.corrcoef(df.yearago_index[8:], df.target_mean[8:])[0,1])
for d in val_days:
    w=(d+8)//7
    print("val day", d, "week", w, "yearago_index:", round(seasonal_index(w),3))

# ---- cell ----
import numpy as np, pandas as pd

def seasonal_fn(view, snapshot_day):
    d = snapshot_day
    keys = view.households.index if isinstance(view.households, pd.DataFrame) else pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.household_key.isin(keys)]
    out = pd.DataFrame(index=keys)
    # tenure per household (first purchase day)
    first = tx.groupby('household_key').day.min()
    tenure = d - first
    # year-ago same-window: days d-363 .. d-336 (future window shifted -364)
    ya = tx[(tx.day > d-364) & (tx.day <= d-336)]
    p13 = ya.groupby('household_key').sales_value.sum().reindex(keys).fillna(0.0)
    ok13 = (tenure >= 364)  # full window observed
    p13 = p13.where(ok13)
    out['p13'] = p13
    # year-ago window one block earlier (stability): d-391..d-364
    ya2 = tx[(tx.day > d-392) & (tx.day <= d-364)]
    p14 = ya2.groupby('household_key').sales_value.sum().reindex(keys).fillna(0.0).where(tenure >= 392)
    out['p13_avg2'] = ((p13.fillna(0)+p14.fillna(0))/2).where(ok13)
    # trailing year-ago 4 weeks: d-391..d-364 is p14; trailing relative to ya window is d-391..d-364
    # ratio: year-ago next-window vs year-ago trailing-window (p14)
    out['p13_ratio'] = p13 / p14.replace(0, np.nan)
    # ratio to current level
    t28 = tx[tx.day > d-28]
    s28 = t28.groupby('household_key').sales_value.sum().reindex(keys).fillna(0.0)
    out['p13_div_s28'] = p13 / s28.replace(0, np.nan)
    out['p13_zero'] = (p13.fillna(0) <= 1.0).astype(float).where(ok13)
    out['tenure'] = tenure
    return out

feats = build_features(seasonal_fn)
print("built:", feats.shape)
print("p13 non-NaN share by snapshot_day:")
print(feats.groupby('snapshot_day').p13.apply(lambda s: s.notna().mean()).round(2).to_string())
save_table(feats.reset_index(), 'season_v1')

# proxy test: E003 base + seasonal block, ridge, temporal val=431 and val=403
base = load_saved('e005_full_plus_mix.parquet')
mix_cols = ['p_GROCERY','p_DRUG GM','p_PRODUCE','p_COSMETICS','p_NUTRITION','p_MEAT','p_MEAT-PCKGD','p_DELI','p_PASTRY','p_FLORAL','p_SEAFOOD-PCKGD','p_MISC. TRANS.','p_SPIRITS','p_SEAFOOD','p_other','p_private','unit_price','n_prod84','dow_entropy','modal_dow','zero_w12','wk_cv']
e3 = base.drop(columns=[c for c in mix_cols if c in base.columns])
print("E003 feats:", e3.shape[1]-2)
tt = train_targets()
m = tt.merge(e3, on=['household_key','snapshot_day'], how='left').merge(feats.reset_index(), on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med
bcols = [c for c in e3.columns if c not in ('household_key','snapshot_day')]
scols = ['p13','p13_avg2','p13_ratio','p13_div_s28','p13_zero','tenure']
Xb,_ = prep(m[bcols]); Xs,_ = prep(m[scols])
Xa = np.hstack([Xb, Xs])
def ridge_mae(X, tr, va, lam=100.0):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Z = (X-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[tr].T@y[tr])
    return float(np.mean(np.abs(y[va]-Zt[va]@w)))
for vd in [431, 403]:
    tr = np.where(m.snapshot_day.isin([x for x in [95,123,151,179,207,235,263,291,319,347,375,403,431] if x!=vd]).values)[0]
    va = np.where((m.snapshot_day==vd).values)[0]
    print(f"val={vd}: base {ridge_mae(Xb,tr,va):.2f}  +season {ridge_mae(Xa,tr,va):.2f}")
# corr of y with p13 among rows with p13
ok = m.p13.notna().values
print("rows with p13:", ok.sum(), "corr(y,p13):", round(float(np.corrcoef(m.p13[ok], y[ok])[0,1]),3),
      "corr(y,spend_84):", round(float(np.corrcoef(m.spend_84[ok], y[ok])[0,1]),3))
# partial given base
tr = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values)[0]
mu, sd = Xb[tr].mean(0), Xb[tr].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[tr].T@Zt[tr] + 100*np.eye(Zt.shape[1]); A[-1,-1]-=100
w = np.linalg.solve(A, Zt[tr].T@y[tr])
res = y - Zt@w
print("partial corr p13|base:", round(float(np.corrcoef(m.p13.fillna(m.p13.median()), res)[0,1]),3))

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
season = load_saved('season_v1.parquet')
season = season.drop(columns=['index'], errors='ignore').rename(columns={'tenure':'yr_tenure'})
mix_cols = ['p_GROCERY','p_DRUG GM','p_PRODUCE','p_COSMETICS','p_NUTRITION','p_MEAT','p_MEAT-PCKGD','p_DELI','p_PASTRY','p_FLORAL','p_SEAFOOD-PCKGD','p_MISC. TRANS.','p_SPIRITS','p_SEAFOOD','p_other','p_private','unit_price','n_prod84','dow_entropy','modal_dow','zero_w12','wk_cv']
e3 = base.drop(columns=[c for c in mix_cols if c in base.columns])
tt = train_targets()
m = tt.merge(e3, on=['household_key','snapshot_day'], how='left').merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
ok = m.p13.notna().values
print("rows with p13:", ok.sum())
print("corr(y,p13):", round(float(np.corrcoef(m.p13[ok], y[ok])[0,1]),3),
      " corr(y,spend_84):", round(float(np.corrcoef(m.spend_84[ok], y[ok])[0,1]),3))
print("corr(y,log1p p13):", round(float(np.corrcoef(np.log1p(m.p13[ok]), y[ok])[0,1]),3))
print("corr(y,log1p spend84):", round(float(np.corrcoef(np.log1p(m.spend_84[ok]), y[ok])[0,1]),3))
print("corr(y,p13_ratio):", round(float(np.corrcoef(m.p13_ratio[ok], y[ok])[0,1]),3))
print("corr(y,p13_div_s28):", round(float(np.corrcoef(m.p13_div_s28[ok], y[ok])[0,1]),3))
print("mean y | p13==0:", round(y[ok & (m.p13.values<=1)],1) if False else round(float(y[ok & (m.p13.values<=1)].mean()),1),
      " mean y | p13>1:", round(float(y[ok & (m.p13.values>1)].mean()),1))

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med
bcols = [c for c in e3.columns if c not in ('household_key','snapshot_day')]
scols = ['p13','p13_avg2','p13_ratio','p13_div_s28','p13_zero','yr_tenure']
Xb,_ = prep(m[bcols]); Xs,_ = prep(m[scols])
Xa = np.hstack([Xb, Xs])
def ridge_mae(X, tr, va, lam=100.0):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Z = (X-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[tr].T@y[tr])
    return float(np.mean(np.abs(y[va]-Zt[va]@w)))
for vd in [431, 403]:
    tr = np.where(m.snapshot_day.isin([x for x in [95,123,151,179,207,235,263,291,319,347,375,403,431] if x!=vd]).values)[0]
    va = np.where((m.snapshot_day==vd).values)[0]
    print(f"val={vd}: base {ridge_mae(Xb,tr,va):.2f}  +season {ridge_mae(Xa,tr,va):.2f}")
tr = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values)[0]
mu, sd = Xb[tr].mean(0), Xb[tr].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[tr].T@Zt[tr] + 100*np.eye(Zt.shape[1]); A[-1,-1]-=100
w = np.linalg.solve(A, Zt[tr].T@y[tr])
res = y - Zt@w
print("partial corr p13|base:", round(float(np.corrcoef(m.p13.fillna(m.p13.median()), res)[0,1]),3))
print("partial corr log1p p13|base:", round(float(np.corrcoef(np.log1p(m.p13.fillna(0)), res)[0,1]),3))

# ---- cell ----
import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
mix_cols = ['p_GROCERY','p_DRUG GM','p_PRODUCE','p_COSMETICS','p_NUTRITION','p_MEAT','p_MEAT-PCKGD','p_DELI','p_PASTRY','p_FLORAL','p_SEAFOOD-PCKGD','p_MISC. TRANS.','p_SPIRITS','p_SEAFOOD','p_other','p_private','unit_price','n_prod84','dow_entropy','modal_dow','zero_w12','wk_cv']
e3 = base.drop(columns=[c for c in mix_cols if c in base.columns]).copy()
print("E003-equiv feats:", e3.shape[1]-2)

out = pd.DataFrame({'household_key': e3.household_key, 'snapshot_day': e3.snapshot_day})
# log1p transforms of main spend levels
for c in ['spend_28','spend_56','spend_84','spend_364','spend_all','x_exp4w']:
    out['log1p_'+c] = np.log1p(e3[c].clip(lower=0))
# per-snapshot winsorized levels (99th pct of cross-section)
g = e3.groupby('snapshot_day')
for c in ['spend_84','spend_28','x_exp4w']:
    cap = g[c].transform(lambda s: s.quantile(0.99))
    out['win_'+c] = np.minimum(e3[c], cap)
# absolute department spends (share x spend_84)
for tag, col in [('gas','x_dep_KIOSK-GAS'),('grocery','x_dep_GROCERY'),('produce','x_dep_PRODUCE'),('meat','x_dep_MEAT'),('drug','x_dep_DRUG GM')]:
    out[tag+'_abs84'] = e3[col].fillna(0) * e3['spend_84'].fillna(0)
# log1p of absolutes
for c in ['gas_abs84','grocery_abs84']:
    out['log1p_'+c] = np.log1p(out[c].clip(lower=0))

e7 = pd.concat([e3.reset_index(drop=True), out.drop(columns=['household_key','snapshot_day']).reset_index(drop=True)], axis=1)
print("E007 table:", e7.shape)
print("new cols:", [c for c in out.columns if c not in ('household_key','snapshot_day')])
print("NaN check:", e7.isna().mean().max())
p = save_table(e7, 'e007_robust')
print("PATH:", p)