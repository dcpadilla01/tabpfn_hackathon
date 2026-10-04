import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e019_final.parquet')
print("shape:", t.shape)
cols = list(t.columns)
print("n cols:", len(cols))
for i in range(0, len(cols), 6):
    print(" | ".join(cols[i:i+6]))
print(t.dtypes.value_counts())
tt = A.train_targets()
print("targets:", tt.shape)
print(A.snapshot_days())

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e019_final.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged:", df.shape)

train_days = A.snapshot_days()['train']; val_days = A.snapshot_days()['validation']
tr = df[df.snapshot_day.isin(train_days)]
va = df[df.snapshot_day.isin(val_days)]
print("train rows:", len(tr), "val rows:", len(va))

feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[feats].astype(float).values; Xva = va[feats].astype(float).values
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)+1e-9
Xtr = np.where(np.isnan(Xtr), mu, (Xtr-mu)/sd); Xva = np.where(np.isnan(Xva), mu, (Xva-mu)/sd)

def fit_ridge(X, y, lam, w=None):
    if w is None:
        return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
    XtX = X.T@(X*w[:,None]); return np.linalg.solve(XtX + lam*np.eye(X.shape[1]), X.T@(w*y))

for lam in [3,10,30,100,300]:
    b = fit_ridge(Xtr, ytr, lam)
    pv = np.clip(Xva@b, 0, None)
    print(f"lam={lam}: val MAE={np.abs(pv-yva).mean():.3f}  trainMAE={np.abs(Xtr@b-ytr).mean():.3f}")


# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

train_days = A.snapshot_days()['train']
tt = A.train_targets()

def prep(path):
    t = A.load_saved(path)
    feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
    return df, feats

def cv_mae(df, feats, lam=30.0, clip=None):
    X = df[feats].astype(float).values.copy()
    if clip is not None:
        lo, hi = clip
        X = np.clip(X, lo, hi)
    y = df.future_spend_4w.values
    days = df.snapshot_day.values
    mu = np.nanmean(X,0); sd = np.nanstd(X,0)+1e-9
    X = np.where(np.isnan(X), mu, (X-mu)/sd)
    errs = []
    for s in train_days:
        m = days != s
        Xt, yt = X[m], y[m]
        b = np.linalg.solve(Xt.T@Xt + lam*np.eye(X.shape[1]), Xt.T@yt)
        pv = np.clip(X[~m]@b, 0, None)
        errs.append(np.abs(pv - y[~m]).mean())
    return float(np.mean(errs)), errs

for path in ['e015_market_ctx.parquet','e018_hinge_prune.parquet','e019_final.parquet']:
    df, feats = prep(path)
    for lam in [30,100]:
        m,_ = cv_mae(df, feats, lam)
        print(f"{path:32s} lam={lam:4d} CV={m:.3f} (nfeat={len(feats)})")


# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

train_days = set(A.snapshot_days()['train'])
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
KEYS = ['household_key','snapshot_day']
feats0 = [c for c in base.columns if c not in KEYS]
y = df.future_spend_4w.values
days = df.snapshot_day.values
is_tr = np.array([d in train_days for d in days])

def stdize(Xtr, Xall):
    mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)+1e-9
    Z = np.where(np.isnan(Xall), mu, Xall)
    return (Z-mu)/sd

def cv(Xall, lams=(30,100,300)):
    out=[]
    Xtr_raw = Xall[is_tr]
    for lam in lams:
        errs=[]
        Xs = stdize(Xtr_raw, Xall)
        for s in sorted(train_days):
            m = days!=s
            b = np.linalg.solve(Xs[m].T@Xs[m] + lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
            pv = np.clip(Xs[~m]@b, 0, None)
            errs.append(np.abs(pv-y[~m]).mean())
        out.append((lam, np.mean(errs)))
    return out

F = df[feats0].astype(float).values
print("BASE:", [(l, round(m,3)) for l,m in cv(F)])

# C1/C2 winsorize
for lo,hi in [(0.01,0.99),(0.005,0.995),(0.02,0.98)]:
    q1 = np.nanquantile(F[is_tr], lo, axis=0); q2 = np.nanquantile(F[is_tr], hi, axis=0)
    W = np.clip(F, q1, q2)
    print(f"WINS{lo}: ", [(l, round(m,3)) for l,m in cv(W)])


# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
KEYS = ['household_key','snapshot_day']
feats0 = [c for c in base.columns if c not in KEYS]
y = df.future_spend_4w.values; days = df.snapshot_day.values

F = df[feats0].astype(float).values
nan_frac = np.isnan(F).mean(0)
order = np.argsort(-nan_frac)
print("cols with NaN>0:", (nan_frac>0).sum(), "of", len(feats0))
for i in order[:25]:
    if nan_frac[i]>0: print(f"  {feats0[i]:22s} {nan_frac[i]:.3f}")

def cv(Xall, fill='quirk', lams=(100,300)):
    out=[]
    for lam in lams:
        errs=[]
        if fill=='quirk':
            mu=np.nanmean(Xall,0); sd=np.nanstd(Xall,0)+1e-9
            Xs=np.where(np.isnan(Xall), mu, (Xall-mu)/sd)
        else:
            mu=np.nanmean(Xall,0); sd=np.nanstd(Xall,0)+1e-9
            Xs=(np.where(np.isnan(Xall), mu, Xall)-mu)/sd
        for s in train_days:
            m=days!=s
            b=np.linalg.solve(Xs[m].T@Xs[m]+lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
            errs.append(np.abs(np.clip(Xs[~m]@b,0,None)-y[~m]).mean())
        out.append((lam, round(float(np.mean(errs)),3)))
    return out

print("quirk:", cv(F,'quirk'))
print("meanfill:", cv(F,'mean'))

# 0-fill + explicit indicators for cols with nan_frac>0.5%
ind_cols=[feats0[i] for i in range(len(feats0)) if 0.005<nan_frac[i]]
I = np.isnan(F[:, [feats0.index(c) for c in ind_cols]]).astype(float)
F2 = np.where(np.isnan(F), 0.0, F)  # careful: 0 in RAW units, then standardize
X2 = np.hstack([F2, I])
print(f"meanfill+ind ({I.shape[1]} inds):", cv(X2,'mean'))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
KEYS = ['household_key','snapshot_day']
feats0 = [c for c in base.columns if c not in KEYS]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
has_nan = np.isnan(F).any(1)
print("rows with any NaN:", has_nan.mean().round(3), " mean y:", y[has_nan].mean().round(1), " vs no-NaN:", y[~has_nan].mean().round(1), " overall:", y.mean().round(1))
# y distribution by NaN status
for nm, m in [('nan',has_nan),('non',~has_nan)]:
    yy=y[m]; print(nm, "p50/75/90:", np.percentile(yy,[50,75,90]).round(1), "frac y=0:", (yy==0).mean().round(3))

mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
def fit_pred(Xs):
    preds=np.zeros(len(y))
    for s in train_days:
        m=days!=s
        b=np.linalg.solve(Xs[m].T@Xs[m]+100*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        preds[~m]=np.clip(Xs[~m]@b,0,None)
    return preds
Q=np.where(np.isnan(F), mu, (F-mu)/sd); pq=fit_pred(Q)
M=(np.where(np.isnan(F), mu, F)-mu)/sd; pm=fit_pred(M)
for nm,m in [('nan',has_nan),('non',~has_nan)]:
    print(nm, "MAE quirk:", np.abs(pq[m]-y[m]).mean().round(2), " MAE meanfill:", np.abs(pm[m]-y[m]).mean().round(2),
          " mean pred q/m:", pq[m].mean().round(1), pm[m].mean().round(1), " mean y:", y[m].mean().round(1))
print("overall corr(pq,pm):", np.corrcoef(pq,pm)[0,1].round(3))
# where does quirk win? error by y-quantile
qb=np.q= pd.qcut(y, 5, duplicates='drop')
print(pd.DataFrame({'y_bin':qb,'eq':np.abs(pq-y),'em':np.abs(pm-y)}).groupby('y_bin',observed=True)[['eq','em']].mean().round(2))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values
F = df[feats0].astype(float).values
mu=np.nanmean(F,0); sd=np.nanstd(F,0)
print("any sd==0:", (sd==0).sum(), " any mu nan:", np.isnan(mu).sum(), " any sd nan:", np.isnan(sd).sum())
M=(np.where(np.isnan(F), mu, F)-mu)/sd
print("M stats: mean", M.mean().round(3), "std", M.std().round(3), "max|.|", np.abs(M).max().round(2))
print("any nan in M:", np.isnan(M).sum(), " any inf:", np.isinf(M).sum())
# single-feature check
i = feats0.index('spend_84')
print("corr(spend_84, y):", np.corrcoef(np.nan_to_num(F[:,i]), y)[0,1].round(3))
b1 = np.linalg.solve(M[:,[i]].T@M[:,[i]]+1, M[:,[i]].T@y)
print("1-feat coef:", b1.round(3), " pred corr:", np.corrcoef(M[:,i]@b1, y)[0,1].round(3))
# full solve diagnostics
XtX = M.T@M
print("XtX diag range:", np.diag(XtX).min().round(3), np.diag(XtX).max().round(3))
b = np.linalg.solve(XtX+100*np.eye(M.shape[1]), M.T@y)
print("coef norm:", np.linalg.norm(b).round(3), " pred mean:", (M@b).mean().round(2), " y mean:", y.mean().round(2))
# maybe y has NaN?
print("y nan:", np.isnan(y).sum(), " y stats:", np.nanmin(y), np.nanmax(y), y.mean().round(1))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values
F = df[feats0].astype(float).values
mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
M=(np.where(np.isnan(F), mu, F)-mu)/sd
XtX = M.T@M; Xty = M.T@y
for lam in [100, 300, 1000]:
    b = np.linalg.solve(XtX+lam*np.eye(M.shape[1]), Xty)
    p = M@b
    print(f"lam={lam}: pred mean={p.mean():.2f} pred std={p.std():.2f} corr={np.corrcoef(p,y)[0,1]:.3f} MAE={np.abs(np.clip(p,0,None)-y).mean():.2f}")
print("y: mean", y.mean().round(1), "std", y.std().round(1))
# check per-column scale of XtX diag
print("XtX diag min/med:", np.diag(XtX).min().round(2), np.median(np.diag(XtX)).round(2))
# condition-ish: top eigenvalue
ev = np.linalg.eigvalsh(XtX)
print("eig min/max:", ev.min().round(3), ev.max().round(1))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values

def loso(Xs, lam, tag):
    errs=[]; pms=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean()); pms.append(p.mean())
    print(f"{tag} lam={lam}: folds=" + " ".join(f"{e:.0f}" for e in errs) + f"  mean={np.mean(errs):.2f}  predmeans={np.round(pms,0)}")

mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
Q=np.where(np.isnan(F), mu, (F-mu)/sd)
M=(np.where(np.isnan(F), mu, F)-mu)/sd
for lam in [100, 1000, 3000, 10000]:
    loso(Q, lam, "quirk   ")
    loso(M, lam, "meanfill")

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values

def loso(Xs, lam, tag):
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean())
    return np.mean(errs)

mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
one = np.ones((len(y),1))
Mint = np.hstack([(np.where(np.isnan(F), mu, F)-mu)/sd, one])
Qint = np.hstack([np.where(np.isnan(F), mu, (F-mu)/sd), one])
for lam in [100, 300, 1000, 3000]:
    print(f"lam={lam}: meanfill+int={loso(Mint,lam,'m'):.2f}  quirk+int={loso(Qint,lam,'q'):.2f}")

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
one = np.ones((len(y),1))
LAM=1000

def loso(Xs, tag):
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+LAM*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean())
    print(f"{tag:34s} {np.mean(errs):.2f}")
    return np.mean(errs)

def prep(Xraw):
    mu=np.nanmean(Xraw,0); sd=np.nanstd(Xraw,0)+1e-9
    return np.hstack([(np.where(np.isnan(Xraw), mu, Xraw)-mu)/sd, one])

loso(prep(F), "BASELINE e019 (intercept)")

# C1: winsorize 1/99
q1=np.nanquantile(F,0.01,0); q2=np.nanquantile(F,0.99,0)
loso(prep(np.clip(F,q1,q2)), "C1 winsorize 1/99")

# C2: missing indicators for the 20.8% NaN group (last-basket etc.)
nf = np.isnan(F).mean(0)
sel = (nf>0.05)&(nf<0.5)
I = np.isnan(F[:,sel]).astype(float)
X2 = np.hstack([np.where(np.isnan(F),0,F), I, one])
mu=np.nanmean(X2,0); sd=np.nanstd(X2,0)+1e-9
X2s=(X2-mu)/sd
loso(X2s, f"C2 zero-fill + {I.shape[1]} indicators")

# C3: extra hinges on more spend sources (6 quantiles each)
extra_src = ['spend_168','spend_364','spend_7','spend_14','spend_21','wk_avg_8','spend_life','spend_rate_life','spend_112','spend_252']
H=[]
for s_ in extra_src:
    v = df[s_].astype(float).values
    qs = np.nanquantile(v, [0.1,0.25,0.5,0.75,0.9,0.97])
    for q in qs:
        H.append(np.maximum(0.0, np.log1p(np.maximum(v,0)) - np.log1p(max(q,0))))
H=np.array(H).T
loso(prep(np.hstack([F,H])), f"C3 +{H.shape[1]} extra hinges")

# C4: interactions of top-8 raw features with demo ordinals
top=['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','trips_84','spend_364']
demo=['size_ord','income_ord','age_ord','grp_ord','kid_ord']
Inter=[]
for t_ in top:
    v=df[t_].astype(float).values
    for d_ in demo:
        Inter.append(v*df[d_].astype(float).values)
Inter=np.array(Inter).T
loso(prep(np.hstack([F,Inter])), f"C4 +{Inter.shape[1]} demo interactions")

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
LAM=1000
def loso(Xs, tag):
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+LAM*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean())
    print(f"{tag:38s} {np.mean(errs):.2f}")
    return np.mean(errs)
def prep(Xraw):
    mu=np.nanmean(Xraw,0); sd=np.nanstd(Xraw,0)+1e-9
    return np.hstack([(np.where(np.isnan(Xraw), mu, Xraw)-mu)/sd, np.ones((len(Xraw),1))])

# market-level forward proxy per snapshot
mkt_fwd = df.groupby('snapshot_day').fwd28_mean.transform(lambda s: s.mean())
mkt_fwd84 = df.groupby('snapshot_day').wk_avg_84.transform(lambda s: s.mean())
Fm = np.hstack([F, mkt_fwd.values[:,None], mkt_fwd84.values[:,None]])
loso(prep(Fm), "D1 +mkt_fwd, mkt_fwd84")

# prune: full-train ridge, keep top-K by |std coef|
mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
Xs=np.hstack([(np.where(np.isnan(F),mu,F)-mu)/sd, np.ones((len(F),1))])
b_full = np.linalg.solve(Xs[days!=431].T@Xs[days!=431]+LAM*np.eye(Xs.shape[1]), Xs[days!=431].T@y[days!=431])
imp = np.abs(b_full[:-1])
order = np.argsort(-imp)
for K in [60, 100, 150, 200]:
    keep = order[:K]
    loso(prep(F[:, keep]), f"D2 prune top-{K}")

# rank-transform key spend features (within snapshot)
key = ['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','spend_364','wk_avg_4']
R = np.hstack([df.groupby('snapshot_day')[k].rank(pct=True).values[:,None] for k in key])
loso(prep(np.hstack([F, R])), "D3 +8 pct-rank features")
loso(prep(np.hstack([F[:, order[:200]], R])), "D4 top-200 + ranks")

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
LAM=1000
def loso(Xraw, tag):
    mu=np.nanmean(Xraw,0); sd=np.nanstd(Xraw,0)+1e-9
    Xs=np.hstack([(np.where(np.isnan(Xraw),mu,Xraw)-mu)/sd, np.ones((len(Xraw),1))])
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+LAM*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        errs.append(np.abs(np.clip(Xs[~m]@b,0,None)-y[~m]).mean())
    print(f"{tag:40s} {np.mean(errs):.2f}")

mkt_fwd = df.groupby('snapshot_day').fwd28_mean.transform('mean').values
demo=['size_ord','income_ord','age_ord','grp_ord','kid_ord']
top=['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','trips_84','spend_364']
Inter=np.column_stack([df[t_].astype(float).values*df[d_].astype(float).values for t_ in top for d_ in demo])
# recency / zero-streak hinges (zero-spend regime)
rec = df.recency.astype(float).values; zs = df.zero_streak.astype(float).values
Hr = np.column_stack([np.maximum(0, rec-k) for k in (14,28,56,84,112)] +
                     [np.maximum(0, zs-k) for k in (0,1,2,4,8)] +
                     [(rec>28).astype(float), (rec>56).astype(float), (rec>84).astype(float)])
E1 = np.hstack([F, Inter]); loso(E1, "E1: e019 + demo interactions")
E2 = np.hstack([F, Inter, mkt_fwd[:,None]]); loso(E2, "E2: + demo inter + mkt_fwd")
E3 = np.hstack([F, Hr]); loso(E3, "E3: + recency/streak hinges")
E4 = np.hstack([F, Inter, Hr, mkt_fwd[:,None]]); loso(E4, "E4: ALL combined")
E5 = np.hstack([F, Hr, mkt_fwd[:,None]]); loso(E5, "E5: hinges + mkt_fwd")

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A
base = A.load_saved('e019_final.parquet')
df = base.copy()
demo=['size_ord','income_ord','age_ord','grp_ord','kid_ord']
top=['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','trips_84','spend_364']
for t_ in top:
    v = df[t_].astype(float).values
    for d_ in demo:
        df[f'ix2_{t_}__{d_}'] = v * df[d_].astype(float).values
df['mkt_fwd'] = df.groupby('snapshot_day').fwd28_mean.transform('mean').values
print("shape:", df.shape)
p = A.save_table(df, 'e020_inter_mkt.parquet')
print("saved:", p)