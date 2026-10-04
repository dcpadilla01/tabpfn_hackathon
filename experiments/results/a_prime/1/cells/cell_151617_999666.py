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
Xb,yb,db,cols_b = prep(E3)
base_mae = ridge_eval(Xb,yb,db)[0]
print("E003 base offline MAE: %.2f"%base_mae)

Xn,yn,dn,cols_n = prep(NF)
# univariate screen
res=[]
for j,c in enumerate(cols_n):
    mae,lam = ridge_eval(Xn[:,[j]],yn,dn); res.append((mae,c))
res.sort()
print("Top univariate (candidates):")
for mae,c in res[:20]: print("  %.2f %s"%(mae,c))
print("Worst:")
for mae,c in res[-8:]: print("  %.2f %s"%(mae,c))

# incremental: add each candidate (one at a time) to E003 base
print("\nIncremental over E003 (top 20):")
inc=[]
for j,c in enumerate(cols_n):
    Xc = np.hstack([Xb, Xn[:,[j]]])
    mae,lam = ridge_eval(Xc,yb,db); inc.append((mae,c))
inc.sort()
for mae,c in inc[:20]: print("  %.3f %s (delta %+.3f)"%(mae,c,mae-base_mae))