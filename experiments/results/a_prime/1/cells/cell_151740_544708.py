import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
def prep(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float); d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            cols.append(pd.factorize(s)[0].astype(float))
        else:
            cols.append(pd.to_numeric(s,errors='coerce').values.astype(float))
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100),fitmax=347):
    fit=days<=fitmax; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_candidates.parquet')
SE = api.load_saved('nf_seasonal.parquet')
Xb,yb,db,cols_b = prep(E3)
Xn,yn,dn,cols_n = prep(NF)
Xs,ys,ds,cols_s = prep(SE)
base = ridge_eval(Xb,yb,db)[0]
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
Xg = np.hstack([Xb]+[Xn[:,[cols_n.index(c)]] for c in greedy])
g_mae = ridge_eval(Xg,yb,db)[0]
print("base %.3f  greedy6 %.3f"%(base,g_mae))

# 1) day-drift terms
for name, v in [('day_idx', db.astype(float)), ('day_idx/100', db.astype(float)/100)]:
    print("  +%-12s %.3f (delta %+.3f)"%(name, ridge_eval(np.hstack([Xg,v[:,None]]),yb,db)[0], ridge_eval(np.hstack([Xg,v[:,None]]),yb,db)[0]-g_mae))

# 2) seasonal table features
for k,c in enumerate(cols_s):
    i = ridge_eval(np.hstack([Xg,Xs[:,[k]]]),yb,db)[0]
    print("  +%-18s %.3f (delta %+.3f)"%(c,i,i-g_mae))

# 3) per-snapshot standardized spend level: z-score spend_l123_mean within snapshot
E1 = api.load_saved('e001_txhist.parquet')
m = t.merge(E1[['household_key','snapshot_day','spend_l123_mean']], on=KEY, how='left')
v = m.spend_l123_mean.values.astype(float)
z = np.empty_like(v)
for s in np.unique(m.snapshot_day.values):
    idx = m.snapshot_day.values==s
    mu = np.nanmean(v[idx]); sd = np.nanstd(v[idx]) or 1
    z[idx] = (v[idx]-mu)/sd
print("  +zspend123      %.3f (delta %+.3f)"%(ridge_eval(np.hstack([Xg,z[:,None]]),yb,db)[0], ridge_eval(np.hstack([Xg,z[:,None]]),yb,db)[0]-g_mae))

# 4) rank-within-snapshot of spend level
r = np.empty_like(v)
for s in np.unique(m.snapshot_day.values):
    idx = m.snapshot_day.values==s
    r[idx] = pd.Series(v[idx]).rank(pct=True).values
print("  +rank_spend123  %.3f (delta %+.3f)"%(ridge_eval(np.hstack([Xg,r[:,None]]),yb,db)[0], ridge_eval(np.hstack([Xg,r[:,None]]),yb,db)[0]-g_mae))

# 5) full package test: greedy6 + hinges + zspend + rank
act = (t.merge(E1[['household_key','snapshot_day','zero_recent']], on=KEY, how='left').zero_recent==0).astype(float).values
s123 = m.spend_l123_mean.values.astype(float)
H = np.column_stack([s123*act, s123*(1-act), np.power(1+s123,0.9)*act, np.power(1+s123,0.9)*(1-act)])
Xfull = np.hstack([Xg, H, z[:,None], r[:,None]])
print("FULL package: %.3f (delta %+.3f)"%(ridge_eval(Xfull,yb,db)[0], ridge_eval(Xfull,yb,db)[0]-g_mae))