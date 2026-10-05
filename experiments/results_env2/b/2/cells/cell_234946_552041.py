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
