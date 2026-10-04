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
E1 = api.load_saved('e001_txhist.parquet')
NF = api.load_saved('nf_candidates.parquet')
X3,y3,d3,c3 = prep(E3)
X1,y1,d1,c1 = prep(E1)
Xn,yn,dn,cols_n = prep(NF)
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
def add_greedy(X): return np.hstack([X]+[Xn[:,[cols_n.index(c)]] for c in greedy])
m = t.merge(E1[['household_key','snapshot_day','spend_l123_mean','zero_recent']], on=KEY, how='left')
s123 = m.spend_l123_mean.values.astype(float)
act = (m.zero_recent==0).astype(float).values
H = np.column_stack([s123*act, s123*(1-act), np.power(1+s123,0.9)*act, np.power(1+s123,0.9)*(1-act)])
day = d3.astype(float)[:,None]

P = {
 'P1 E003+g6+H+day': np.hstack([add_greedy(X3), H, day]),
 'P2 E001+g6+H+day': np.hstack([add_greedy(X1), H, day]),
 'P3 E003+g6+H':     np.hstack([add_greedy(X3), H]),
 'P4 E003+g6+day':   np.hstack([add_greedy(X3), day]),
 'P5 E003+g6+H+day (no pow90 hinge)': np.hstack([add_greedy(X3), H[:,[0,1]], day]),
}
for k,Xp in P.items():
    mae,lam = ridge_eval(Xp,y3,d3)
    print("%-38s %.3f (lam %g)"%(k,mae,lam))
# also lam grid finer for best
Xp = P['P1 E003+g6+H+day']
mae,lam = ridge_eval(Xp,y3,d3,lams=(1,3,10,30,100,300,1000))
print("P1 finer lam: %.3f (%g)"%(mae,lam))