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
