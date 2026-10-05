import numpy as np, pandas as pd, re
import agent_api as A

e = A.load_saved("e011_table.parquet")
print("e011 shape", e.shape)
cols = [c for c in e.columns if c not in ("household_key","snapshot_day")]
pref = {}
for c in cols:
    p = re.split(r"[_0-9]", c)[0] or c
    pref.setdefault(p, []).append(c)
for p, cs in sorted(pref.items()):
    print(f"{p:12s} n={len(cs):3d} e.g. {cs[:5]}")

tt = A.train_targets()
print("\ntargets", tt.shape)
print(tt.future_spend_4w.describe())

# local target computation from capped snapshot(459), verify against train_targets
v = A.snapshot()
tx = v.transactions
print("tx<=459:", tx.shape, "maxday", tx.day.max())
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
C = np.hstack([np.zeros((piv.shape[0],1)), piv.values.cumsum(1)])
hidx = {h:i for i,h in enumerate(piv.index)}
tt2 = tt.copy()
tt2["i"] = tt2.household_key.map(hidx)
print("missing hh:", tt2.i.isna().sum())
tt2["i"] = tt2.i.astype(int)
t_ = tt2.snapshot_day.values
yloc = C[tt2.i, t_+28] - C[tt2.i, t_]
print("max |local - train_target| =", np.abs(yloc - tt2.future_spend_4w.values).max())

y = tt2.future_spend_4w.values
print("\nMAE global mean:", np.abs(y-y.mean()).mean())
w1 = C[tt2.i, t_] - C[tt2.i, t_-28]
print("MAE pred=w1(trailing28):", np.abs(y-w1).mean(), "corr(y,w1)=", np.corrcoef(y,w1)[0,1])
# past targets w2..w14 mean as predictor
K = 14
Wm = np.full((len(tt2), K), np.nan)
for k in range(1, K+1):
    end = t_ - 28*(k-1); start = end-27
    ok = start >= 1
    Wm[ok, k-1] = C[tt2.i[ok], end[ok]] - C[tt2.i[ok], start[ok]-1]
past = np.nanmean(Wm[:,1:], axis=1)
print("MAE pred=mean(w2..w14):", np.abs(y-past).mean())
# global ratio per snapshot (aligned windows, past only) -> pred = w1*global_ratio
preds = np.full(len(tt2), np.nan)
for j,t in enumerate(sorted(tt2.snapshot_day.unique())):
    m = tt2.snapshot_day.values==t
    ds = np.array([t-28*k for k in range(1, 14) if t-28*k >= 28])
    f = C[:, ds+28] - C[:, ds]
    b = C[:, ds] - C[:, ds-28]
    r = np.where(b>0, f/np.maximum(b,1e-9), np.nan)
    gm = np.nanmean(np.clip(r,0,25))
    preds[m] = (w1[m]) * gm
print("MAE pred=w1*global_aligned_ratio:", np.abs(y-preds).mean())
print("y quantiles:", np.quantile(y,[.5,.75,.9,.95,.99]).round(1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot()
tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 461)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
tr = tx.groupby(["household_key","day"]).basket_id.nunique().unstack(fill_value=0.0).reindex(index=piv.index, columns=days, fill_value=0.0)
CT = np.zeros((H, 461)); CT[:,1:] = tr.values.cumsum(1)
def sp(a,b): return C[:, np.clip(b+1,0,460)] - C[:, np.clip(a,0,460)]
def tp(a,b): return CT[:, np.clip(b+1,0,460)] - CT[:, np.clip(a,0,460)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 460 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values
y = tt.future_spend_4w.values
w1 = sp(t_-27, t_)[ii]

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 11))
        for j in range(8):
            F[:,j] = np.log1p(sp(s-7*(j+1)+1, s-7*j))
        F[:,8] = np.log1p(tp(s-55, s))
        F[:,9] = np.log1p(sp(s-27, s))
        F[:,10] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28)
        w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok]); Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok],20.0),0,4)); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9
    Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan); out_rat = np.full(H, np.nan); out_k10 = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 11))
    for j in range(8): Ft[:,j] = np.log1p(sp(t-7*(j+1)+1, t-7*j))[tgt]
    Ft[:,8] = np.log1p(tp(t-55, t))[tgt]
    Ft[:,9] = np.log1p(sp(t-27, t))[tgt]
    Ft[:,10] = np.clip(t - last_le(t), 0, 56)/56.0
    Zt = (Ft-mu)/sd
    D2 = (Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T
    D = np.sqrt(np.maximum(D2, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy()
        selfmask = Gs == h
        d[selfmask] = np.inf
        order = np.argsort(d)[:30]
        dd = d[order]; yy = Ys[order]; rr = Rs[order]
        wgt = 1.0/(dd+0.5)
        out_abs[h] = (wgt*yy).sum()/wgt.sum()
        out_rat[h] = w1[h]*np.clip((wgt*rr).sum()/wgt.sum(), 0, 4)
        w2_ = 1.0/(dd[:10]+0.5)
        out_k10[h] = (w2_*yy[:10]).sum()/w2_.sum()
    return out_abs, out_rat, out_k10, tgt

abs_all = np.full(len(tt), np.nan); rat_all = np.full(len(tt), np.nan); k10_all = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_, r_, k_, tgt = twin_for_day(int(t))
    m = t_==t
    abs_all[m] = a_[ii[m]]; rat_all[m] = r_[ii[m]]; k10_all[m] = k_[ii[m]]
print("twin computed", round(time.time()-t0,1), "nan frac", round(np.isnan(abs_all).mean(),3))

def mae(p): 
    ok = ~np.isnan(p); return np.abs(y[ok]-p[ok]).mean()
print("MAE w1           :", round(np.abs(y-w1).mean(),3))
print("MAE twin_abs k30 :", round(mae(abs_all),3))
print("MAE twin_abs k10 :", round(mae(k10_all),3))
print("MAE twin_ratio   :", round(mae(rat_all),3))
print("rows w1==0:", (w1==0).sum(), "mean y:", round(y[w1==0].mean(),2), "| w1>0 mean y:", round(y[w1>0].mean(),2))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot()
tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values
hidx = {h:i for i,h in enumerate(hh)}
tr = tx.groupby(["household_key","day"]).basket_id.nunique().unstack(fill_value=0.0).reindex(index=piv.index, columns=days, fill_value=0.0)
CT = np.zeros((H, 460)); CT[:,1:] = tr.values
def sp(a,b): return C[:, np.clip(b+1,0,459)] - C[:, np.clip(a,0,459)]
def tp(a,b): return CT[:, np.clip(b+1,0,459)] - CT[:, np.clip(a,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values
y = tt.future_spend_4w.values
w1 = sp(t_-27, t_)[ii]
print("target check:", round(np.abs((C[ii, t_+28]-C[ii, t_]) - y).max(),9))

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 11))
        for j in range(8):
            F[:,j] = np.log1p(sp(s-7*(j+1)+1, s-7*j))
        F[:,8] = np.log1p(tp(s-55, s))
        F[:,9] = np.log1p(sp(s-27, s))
        F[:,10] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28)
        w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok]); Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok],20.0),0,4)); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9
    Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan); out_rat = np.full(H, np.nan); out_k10 = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 11))
    for j in range(8): Ft[:,j] = np.log1p(sp(t-7*(j+1)+1, t-7*j))[tgt]
    Ft[:,8] = np.log1p(tp(t-55, t))[tgt]
    Ft[:,9] = np.log1p(sp(t-27, t))[tgt]
    Ft[:,10] = np.clip(t - last_le(t), 0, 56)/56.0
    Zt = (Ft-mu)/sd
    D2 = (Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T
    D = np.sqrt(np.maximum(D2, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy()
        d[Gs == h] = np.inf
        order = np.argsort(d)[:30]
        dd = d[order]; yy = Ys[order]; rr = Rs[order]
        wgt = 1.0/(dd+0.5)
        out_abs[h] = (wgt*yy).sum()/wgt.sum()
        out_rat[h] = w1[h]*np.clip((wgt*rr).sum()/wgt.sum(), 0, 4)
        w2_ = 1.0/(dd[:10]+0.5)
        out_k10[h] = (w2_*yy[:10]).sum()/w2_.sum()
    return out_abs, out_rat, out_k10, tgt

abs_all = np.full(len(tt), np.nan); rat_all = np.full(len(tt), np.nan); k10_all = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_, r_, k_, tgt = twin_for_day(int(t))
    m = t_==t
    abs_all[m] = a_[ii[m]]; rat_all[m] = r_[ii[m]]; k10_all[m] = k_[ii[m]]
print("twin computed", round(time.time()-t0,1), "nan frac", round(np.isnan(abs_all).mean(),3))

def mae(p): 
    ok = ~np.isnan(p); return np.abs(y[ok]-p[ok]).mean()
print("MAE w1           :", round(np.abs(y-w1).mean(),3))
print("MAE twin_abs k30 :", round(mae(abs_all),3))
print("MAE twin_abs k10 :", round(mae(k10_all),3))
print("MAE twin_ratio   :", round(mae(rat_all),3))
print("rows w1==0:", (w1==0).sum(), "mean y:", round(y[w1==0].mean(),2), "| w1>0 mean y:", round(y[w1>0].mean(),2))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot()
tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values
hidx = {h:i for i,h in enumerate(hh)}
tr = tx.groupby(["household_key","day"]).basket_id.nunique().unstack(fill_value=0.0).reindex(index=piv.index, columns=days, fill_value=0.0)
CT = np.zeros((H, 460)); CT[:,1:] = tr.values
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def tp(a,b): return CT[:, np.clip(b,0,459)] - CT[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values
y = tt.future_spend_4w.values
w1 = sp(t_-27, t_)[ii]
print("target check:", round(np.abs((C[ii, t_+28]-C[ii, t_]) - y).max(),9))

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 11))
        for j in range(8):
            F[:,j] = np.log1p(sp(s-7*(j+1)+1, s-7*j))
        F[:,8] = np.log1p(tp(s-55, s))
        F[:,9] = np.log1p(sp(s-27, s))
        F[:,10] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28)
        w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok]); Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok],20.0),0,4)); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9
    Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan); out_rat = np.full(H, np.nan); out_k10 = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 11))
    for j in range(8): Ft[:,j] = np.log1p(sp(t-7*(j+1)+1, t-7*j))[tgt]
    Ft[:,8] = np.log1p(tp(t-55, t))[tgt]
    Ft[:,9] = np.log1p(sp(t-27, t))[tgt]
    Ft[:,10] = (np.clip(t - last_le(t), 0, 56)/56.0)[tgt]
    Zt = (Ft-mu)/sd
    D2 = (Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T
    D = np.sqrt(np.maximum(D2, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy()
        d[Gs == h] = np.inf
        order = np.argsort(d)[:30]
        dd = d[order]; yy = Ys[order]; rr = Rs[order]
        wgt = 1.0/(dd+0.5)
        out_abs[h] = (wgt*yy).sum()/wgt.sum()
        out_rat[h] = w1[h]*np.clip((wgt*rr).sum()/wgt.sum(), 0, 4)
        w2_ = 1.0/(dd[:10]+0.5)
        out_k10[h] = (w2_*yy[:10]).sum()/w2_.sum()
    return out_abs, out_rat, out_k10, tgt

abs_all = np.full(len(tt), np.nan); rat_all = np.full(len(tt), np.nan); k10_all = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_, r_, k_, tgt = twin_for_day(int(t))
    m = t_==t
    abs_all[m] = a_[ii[m]]; rat_all[m] = r_[ii[m]]; k10_all[m] = k_[ii[m]]
print("twin computed", round(time.time()-t0,1), "nan frac", round(np.isnan(abs_all).mean(),3))

def mae(p): 
    ok = ~np.isnan(p); return np.abs(y[ok]-p[ok]).mean()
print("MAE w1           :", round(np.abs(y-w1).mean(),3))
print("MAE twin_abs k30 :", round(mae(abs_all),3))
print("MAE twin_abs k10 :", round(mae(k10_all),3))
print("MAE twin_ratio   :", round(mae(rat_all),3))
print("rows w1==0:", (w1==0).sum(), "mean y:", round(y[w1==0].mean(),2), "| w1>0 mean y:", round(y[w1>0].mean(),2))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot()
tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
tr = tx.groupby(["household_key","day"]).basket_id.nunique().unstack(fill_value=0.0).reindex(index=piv.index, columns=days, fill_value=0.0)
CT = np.zeros((H, 460)); CT[:,1:] = tr.values.cumsum(1)
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def tp(a,b): return CT[:, np.clip(b,0,459)] - CT[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values
y = tt.future_spend_4w.values
w1 = sp(t_-27, t_)[ii]
print("target check:", round(np.abs((C[ii, t_+28]-C[ii, t_]) - y).max(),9))

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 11))
        for j in range(8):
            F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j),0))
        F[:,8] = np.log1p(tp(s-55, s))
        F[:,9] = np.log1p(np.maximum(sp(s-27, s),0))
        F[:,10] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28)
        w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok]); Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok],20.0),0,4)); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9
    Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan); out_rat = np.full(H, np.nan); out_k10 = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 11))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j),0))[tgt]
    Ft[:,8] = np.log1p(tp(t-55, t))[tgt]
    Ft[:,9] = np.log1p(np.maximum(sp(t-27, t),0))[tgt]
    Ft[:,10] = (np.clip(t - last_le(t), 0, 56)/56.0)[tgt]
    Zt = (Ft-mu)/sd
    D2 = (Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T
    D = np.sqrt(np.maximum(D2, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy()
        d[Gs == h] = np.inf
        order = np.argsort(d)[:30]
        dd = d[order]; yy = Ys[order]; rr = Rs[order]
        wgt = 1.0/(dd+0.5)
        out_abs[h] = float((wgt*yy).sum()/wgt.sum())
        out_rat[h] = float(w1[h]*np.clip((wgt*rr).sum()/wgt.sum(), 0, 4))
        w2_ = 1.0/(dd[:10]+0.5)
        out_k10[h] = float((w2_*yy[:10]).sum()/w2_.sum())
    return out_abs, out_rat, out_k10, tgt

abs_all = np.full(len(tt), np.nan); rat_all = np.full(len(tt), np.nan); k10_all = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_, r_, k_, tgt = twin_for_day(int(t))
    m = t_==t
    abs_all[m] = a_[ii[m]]; rat_all[m] = r_[ii[m]]; k10_all[m] = k_[ii[m]]
print("twin computed", round(time.time()-t0,1), "nan frac", round(np.isnan(abs_all).mean(),3))

def mae(p): 
    ok = ~np.isnan(p); return np.abs(y[ok]-p[ok]).mean()
print("MAE w1           :", round(np.abs(y-w1).mean(),3))
print("MAE twin_abs k30 :", round(mae(abs_all),3))
print("MAE twin_abs k10 :", round(mae(k10_all),3))
print("MAE twin_ratio   :", round(mae(rat_all),3))
print("rows w1==0:", (w1==0).sum(), "mean y:", round(y[w1==0].mean(),2), "| w1>0 mean y:", round(y[w1>0].mean(),2))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot()
tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
tr = tx.groupby(["household_key","day"]).basket_id.nunique().unstack(fill_value=0.0).reindex(index=piv.index, columns=days, fill_value=0.0)
CT = np.zeros((H, 460)); CT[:,1:] = tr.values.cumsum(1)
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def tp(a,b): return CT[:, np.clip(b,0,459)] - CT[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values
y = tt.future_spend_4w.values
w1 = C[ii, t_] - C[ii, np.clip(t_-28,0,459)]
print("target check:", round(np.abs((C[ii, t_+28]-C[ii, t_]) - y).max(),9), "| w1 shape", w1.shape)

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 11))
        for j in range(8):
            F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j),0))
        F[:,8] = np.log1p(tp(s-55, s))
        F[:,9] = np.log1p(np.maximum(sp(s-27, s),0))
        F[:,10] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28)
        w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok]); Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok],20.0),0,4)); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9
    Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan); out_rat = np.full(H, np.nan); out_k10 = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 11))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j),0))[tgt]
    Ft[:,8] = np.log1p(tp(t-55, t))[tgt]
    Ft[:,9] = np.log1p(np.maximum(sp(t-27, t),0))[tgt]
    Ft[:,10] = (np.clip(t - last_le(t), 0, 56)/56.0)[tgt]
    Zt = (Ft-mu)/sd
    D2 = (Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T
    D = np.sqrt(np.maximum(D2, 0))
    w1t = sp(t-27, t)
    for a, h in enumerate(tgt):
        d = D[a].copy()
        d[Gs == h] = np.inf
        order = np.argsort(d)[:30]
        dd = d[order]; yy = Ys[order]; rr = Rs[order]
        wgt = 1.0/(dd+0.5)
        out_abs[h] = (wgt*yy).sum()/wgt.sum()
        out_rat[h] = w1t[h]*np.clip((wgt*rr).sum()/wgt.sum(), 0, 4)
        w2_ = 1.0/(dd[:10]+0.5)
        out_k10[h] = (w2_*yy[:10]).sum()/w2_.sum()
    return out_abs, out_rat, out_k10, tgt

abs_all = np.full(len(tt), np.nan); rat_all = np.full(len(tt), np.nan); k10_all = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_, r_, k_, tgt = twin_for_day(int(t))
    m = t_==t
    abs_all[m] = a_[ii[m]]; rat_all[m] = r_[ii[m]]; k10_all[m] = k_[ii[m]]
print("twin computed", round(time.time()-t0,1), "nan frac", round(np.isnan(abs_all).mean(),3))

def mae(p): 
    ok = ~np.isnan(p); return np.abs(y[ok]-p[ok]).mean()
print("MAE w1           :", round(np.abs(y-w1).mean(),3))
print("MAE twin_abs k30 :", round(mae(abs_all),3))
print("MAE twin_abs k10 :", round(mae(k10_all),3))
print("MAE twin_ratio   :", round(mae(rat_all),3))
print("rows w1==0:", (w1==0).sum(), "mean y:", round(y[w1==0].mean(),2), "| w1>0 mean y:", round(y[w1>0].mean(),2))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot(); tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values; y = tt.future_spend_4w.values
w1 = C[ii, t_] - C[ii, t_-28]
w2 = C[ii, t_-28] - C[ii, t_-56]
usual = (C[ii, t_] - C[ii, t_-84])/3.0
ya = C[ii, t_-364] - C[ii, t_-392]
y1ago = C[ii, t_-336] - C[ii, t_-364]
sc_f = np.clip((C[ii, t_-336] - C[ii, t_-364])/np.maximum(C[ii, t_-364] - C[ii, t_-392], 20.0), 0, 4)

# rebuild twin quickly (same as before)
def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 11))
        for j in range(8): F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j),0))
        F[:,9] = np.log1p(np.maximum(sp(s-27, s),0))
        F[:,10] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28); w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok]); Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok],20.0),0,4)); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9; Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 11))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j),0))[tgt]
    Ft[:,9] = np.log1p(np.maximum(sp(t-27, t),0))[tgt]
    Ft[:,10] = (np.clip(t - last_le(t), 0, 56)/56.0)[tgt]
    Zt = (Ft-mu)/sd
    D = np.sqrt(np.maximum((Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy(); d[Gs == h] = np.inf
        order = np.argsort(d)[:30]; wgt = 1.0/(d[order]+0.5)
        out_abs[h] = (wgt*Ys[order]).sum()/wgt.sum()
    return out_abs
twin = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_ = twin_for_day(int(t)); m = t_==t; twin[m] = a_[ii[m]]
print("twin done", round(time.time()-t0,1))

def ridge_fit(Xtr, ytr, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sd; Zb = np.hstack([Z, np.ones((len(Z),1))])
    A_ = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A_[-1,-1] -= lam
    w = np.linalg.solve(A_, Zb.T@ytr)
    return mu, sd, w
def ridge_pred(X, mu, sd, w):
    Z = (X-mu)/sd; Zb = np.hstack([Z, np.ones((len(Z),1))])
    return Zb@w

trm = t_ <= 431; vam = t_ >= 459
def run(feats, names):
    X = np.column_stack([f for f in feats])
    mu, sd, w = ridge_fit(X[trm], y[trm])
    p = ridge_pred(X[vam], mu, sd, w)
    return np.abs(y[vam]-p).mean()

base = [w1, usual, w2, ya, y1ago, sc_f]
print("val MAE base6        :", round(run(base, []),3))
print("val MAE base6+twin   :", round(run(base+[twin], []),3))
print("val MAE base6+twin+r :", round(run(base+[twin, twin*w1], []),3))
# residual correlation
r = y - w1
print("corr(y-w1, twin-w1):", round(np.corrcoef(r, twin-w1)[0,1],4))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot(); tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values; y = tt.future_spend_4w.values
w1 = C[ii, t_] - C[ii, t_-28]
w2 = C[ii, t_-28] - C[ii, t_-56]
usual = (C[ii, t_] - C[ii, t_-84])/3.0
ya = C[ii, t_-364] - C[ii, t_-392]
y1ago = C[ii, t_-336] - C[ii, t_-364]
sc_f = np.clip((C[ii, t_-336] - C[ii, t_-364])/np.maximum(C[ii, t_-364] - C[ii, t_-392], 20.0), 0, 4)

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Gs = [], [], []
    for s in ss:
        F = np.zeros((H, 10))
        for j in range(8): F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j),0))
        F[:,8] = np.log1p(np.maximum(sp(s-27, s),0))
        F[:,9] = np.clip(s - last_le(s), 0, 56)/56.0
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(sp(s+1, s+28)[ok]); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9; Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 10))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j),0))[tgt]
    Ft[:,8] = np.log1p(np.maximum(sp(t-27, t),0))[tgt]
    Ft[:,9] = (np.clip(t - last_le(t), 0, 56)/56.0)[tgt]
    Zt = (Ft-mu)/sd
    D = np.sqrt(np.maximum((Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy(); d[Gs == h] = np.inf
        order = np.argsort(d)[:30]; wgt = 1.0/(d[order]+0.5)
        out_abs[h] = (wgt*Ys[order]).sum()/wgt.sum()
    return out_abs
twin = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_ = twin_for_day(int(t)); m = t_==t; twin[m] = a_[ii[m]]
print("twin done", round(time.time()-t0,1))

def ridge_fit(Xtr, ytr, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sd; Zb = np.hstack([Z, np.ones((len(Z),1))])
    A_ = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A_[-1,-1] -= lam
    w = np.linalg.solve(A_, Zb.T@ytr)
    return mu, sd, w
trm = t_ <= 375; vam = t_ >= 403
def run(feats):
    X = np.column_stack(feats)
    mu, sd, w = ridge_fit(X[trm], y[trm])
    p = ridge_pred = None
    Z = (X[vam]-mu)/sd; Zb = np.hstack([Z, np.ones((vam.sum(),1))])
    return np.abs(y[vam]-Zb@w).mean()

base = [w1, usual, w2, ya, y1ago, sc_f]
print("val MAE base6      :", round(run(base),3))
print("val MAE base6+twin :", round(run(base+[twin]),3))
print("val MAE base6+twin+twin*w1:", round(run(base+[twin, twin*w1]),3))
r = y - w1
print("corr(y-w1, twin-w1):", round(np.corrcoef(r, twin-w1)[0,1],4))
# twin on w1==0 rows
z = w1==0
print("w1==0 rows:", z.sum(), "mean twin there:", round(np.nanmean(twin[z]),2), "mean y:", round(y[z].mean(),2))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot(); tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values; y = tt.future_spend_4w.values
w1 = C[ii, t_] - C[ii, t_-28]
w2 = C[ii, t_-28] - C[ii, t_-56]
usual = (C[ii, t_] - C[ii, t_-84])/3.0
ya = C[ii, t_-364] - C[ii, t_-392]
y1ago = C[ii, t_-336] - C[ii, t_-364]
sc_f = np.clip((C[ii, t_-336] - C[ii, t_-364])/np.maximum(C[ii, t_-364] - C[ii, t_-392], 20.0), 0, 4)

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Gs = [], [], []
    for s in ss:
        F = np.zeros((H, 10))
        for j in range(8): F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j),0))
        F[:,8] = np.log1p(np.maximum(sp(s-27, s),0))
        F[:,9] = np.clip(s - last_le(s), 0, 56)/56.0
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(sp(s+1, s+28)[ok]); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9; Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan)
    tgt = np.where(last_le(t) >= 84)[0]
    Ft = np.zeros((len(tgt), 10))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j),0))[tgt]
    Ft[:,8] = np.log1p(np.maximum(sp(t-27, t),0))[tgt]
    Ft[:,9] = (np.clip(t - last_le(t), 0, 56)/56.0)[tgt]
    Zt = (Ft-mu)/sd
    D = np.sqrt(np.maximum((Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T, 0))
    for a, h in enumerate(tgt):
        d = D[a].copy(); d[Gs == h] = np.inf
        order = np.argsort(d)[:30]; wgt = 1.0/(d[order]+0.5)
        out_abs[h] = (wgt*Ys[order]).sum()/wgt.sum()
    return out_abs
twin_raw = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_ = twin_for_day(int(t)); m = t_==t; twin_raw[m] = a_[ii[m]]
twin = np.where(np.isnan(twin_raw), w1, twin_raw)
print("twin done", round(time.time()-t0,1), "filled:", np.isnan(twin_raw).sum())

def ridge_fit(Xtr, ytr, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sd; Zb = np.hstack([Z, np.ones((len(Z),1))])
    A_ = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A_[-1,-1] -= lam
    w = np.linalg.solve(A_, Zb.T@ytr)
    return mu, sd, w
trm = t_ <= 375; vam = t_ >= 403
def run(feats):
    X = np.column_stack(feats)
    mu, sd, w = ridge_fit(X[trm], y[trm])
    Z = (X[vam]-mu)/sd; Zb = np.hstack([Z, np.ones((vam.sum(),1))])
    return np.abs(y[vam]-Zb@w).mean()

base = [w1, usual, w2, ya, y1ago, sc_f]
print("val MAE base6            :", round(run(base),3))
print("val MAE base6+twin       :", round(run(base+[twin]),3))
print("val MAE base6+twin+twin*w1:", round(run(base+[twin, twin*w1]),3))
r = y - w1
print("corr(y-w1, twin-w1):", round(np.corrcoef(r, twin-w1)[0,1],4))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd, time
import agent_api as A

t0=time.time()
v = A.snapshot(); tx = v.transactions
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
hh = piv.index.to_numpy(); H = len(hh)
C = np.zeros((H, 460)); C[:,1:] = piv.values.cumsum(1)
hidx = {h:i for i,h in enumerate(hh)}
def sp(a,b): return C[:, np.clip(b,0,459)] - C[:, np.clip(a-1,0,459)]
def last_le(s):
    act = piv.values > 0; act[:, s:] = False
    has = act.any(1)
    return np.where(has, 459 - act[:,::-1].argmax(1), 0)

tt = A.train_targets()
ii = tt.household_key.map(hidx).astype(int).values
t_ = tt.snapshot_day.values; y = tt.future_spend_4w.values

def twin_for_day(t):
    ss = [s for s in (t-28, t-56) if s >= 57]
    Xs, Ys, Gs = [], [], []
    for s in ss:
        F = np.zeros((H, 10))
        for j in range(8): F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j),0))
        F[:,8] = np.log1p(np.maximum(sp(s-27, s),0))
        F[:,9] = np.clip(s - last_le(s), 0, 56)/56.0
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(sp(s+1, s+28)[ok]); Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Gs = np.concatenate(Gs)
    mu, sd = Xs.mean(0), Xs.std(0)+1e-9; Zr = (Xs-mu)/sd
    out_abs = np.full(H, np.nan)
    Ft = np.zeros((H, 10))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j),0))
    Ft[:,8] = np.log1p(np.maximum(sp(t-27, t),0))
    Ft[:,9] = np.clip(t - last_le(t), 0, 56)/56.0
    Zt = (Ft-mu)/sd
    D = np.sqrt(np.maximum((Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T, 0))
    for a in range(H):
        d = D[a].copy(); d[Gs == a] = np.inf
        order = np.argsort(d)[:30]; wgt = 1.0/(d[order]+0.5)
        out_abs[a] = (wgt*Ys[order]).sum()/wgt.sum()
    return out_abs
twin = np.full(len(tt), np.nan)
for t in sorted(tt.snapshot_day.unique()):
    a_ = twin_for_day(int(t)); m = t_==t; twin[m] = a_[ii[m]]
print("twin done", round(time.time()-t0,1), "nan:", np.isnan(twin).sum())

# E011 features locally
e = A.load_saved("e011_table.parquet")
feat_cols = [c for c in e.columns if c not in ("household_key","snapshot_day")]
mrg = tt.merge(e, on=["household_key","snapshot_day"], how="left")
Xe = mrg[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
print("E011 X:", Xe.shape)

def ridge_fit(Xtr, ytr, lam=3.0):
    mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)+1e-9
    Z = (Xtr-mu)/sd; Zb = np.hstack([Z, np.ones((len(Z),1))])
    A_ = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A_[-1,-1] -= lam
    w = np.linalg.solve(A_, Zb.T@ytr)
    return mu, sd, w
trm = t_ <= 375; vam = t_ >= 403
def run(cols_extra):
    X = np.hstack([Xe] + [c.reshape(-1,1) for c in cols_extra]) if cols_extra else Xe
    mu, sd, w = ridge_fit(X[trm], y[trm])
    Z = (X[vam]-mu)/sd; Zb = np.hstack([Z, np.ones((vam.sum(),1))])
    return np.abs(y[vam]-Zb@w).mean()
print("local val MAE E011 feats      :", round(run([]),3))
print("local val MAE E011 + twin     :", round(run([twin]),3))
tw2 = np.where(np.isnan(twin), 0.0, twin)
print("local val MAE E011 + twin+tw*w1:", round(run([tw2, tw2*(C[ii,t_]-C[ii,t_-28])]),3))
print("total t", round(time.time()-t0,1))


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

def fn(view, t):
    tx = view.table("transactions")
    piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0)
    piv = piv.reindex(columns=np.arange(1, t+1), fill_value=0.0)
    hh = piv.index.to_numpy(); H = len(hh)
    C = np.zeros((H, t+1)); C[:,1:] = piv.values.cumsum(1)
    def sp(a,b): return C[:, np.clip(b,0,t)] - C[:, np.clip(a-1,0,t)]
    act = piv.values > 0
    def last_le(s):
        a2 = act.copy(); a2[:, s:] = False
        has = a2.any(1)
        return np.where(has, t - a2[:,::-1].argmax(1), 0)
    l_t = last_le(t)
    ss = [t-28] + ([t-56] if t-56 >= 57 else [])
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 10))
        for j in range(8): F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j), 0))
        F[:,8] = np.log1p(np.maximum(sp(s-27, s), 0))
        F[:,9] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28); w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok])
        Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok], 20.0), 0, 4))
        Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu = Xs.mean(0); sd = Xs.std(0) + 1e-9
    Zr = (Xs - mu)/sd
    Ft = np.zeros((H, 10))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j), 0))
    Ft[:,8] = np.log1p(np.maximum(sp(t-27, t), 0))
    Ft[:,9] = np.clip(t - l_t, 0, 56)/56.0
    Zt = (Ft - mu)/sd
    D = np.sqrt(np.maximum((Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T, 0))
    w1t = sp(t-27, t)
    out30 = np.zeros(H); out10 = np.zeros(H); outrat = np.zeros(H)
    for a in range(H):
        d = D[a].copy(); d[Gs == a] = np.inf
        order = np.argsort(d)[:30]
        w30 = 1.0/(d[order] + 0.5)
        out30[a] = (w30*Ys[order]).sum()/w30.sum()
        outrat[a] = w1t[a]*np.clip((w30*Rs[order]).sum()/w30.sum(), 0, 4)
        w10 = 1.0/(d[order[:10]] + 0.5)
        out10[a] = (w10*Ys[order[:10]]).sum()/w10.sum()
    out = pd.DataFrame({"tw_k30": out30, "tw_k10": out10, "tw_ratio": outrat,
                        "tw_diff": out30 - w1t}, index=pd.Index(hh, name="household_key"))
    return out.reindex(view.households)

feats = A.build_features(fn)
print("built:", feats.shape, "| nan:", feats.isna().sum().sum())

e = A.load_saved("e011_table.parquet")
mrg = e.merge(feats.reset_index(), on=["household_key","snapshot_day"], how="left")
print("merged:", mrg.shape, "| tw nan:", mrg[["tw_k30","tw_k10","tw_ratio","tw_diff"]].isna().sum().to_dict())
assert mrg.shape[0] == e.shape[0]
path = A.save_table(mrg, "twin_v1")
print("saved:", path)
