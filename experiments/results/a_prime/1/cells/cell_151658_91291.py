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
E1 = api.load_saved('e001_txhist.parquet')
Xb,yb,db,cols_b = prep(E3)
Xn,yn,dn,cols_n = prep(NF)
base = ridge_eval(Xb,yb,db)[0]

greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
Xg = np.hstack([Xb]+[Xn[:,[cols_n.index(c)]] for c in greedy])
g_mae = ridge_eval(Xg,yb,db)[0]
print("E003+greedy6: %.3f"%g_mae)

# add remaining pool features one at a time on top of greedy
pool2 = ['nspend_l11','nspend_l10','nspend_l9','nspend_l8','nspend_l7','nf_ncv12','nbask_max84','nwstd12','nprods84','nf_nspend28_pow75','newm4','nspend14','ngap_mean','nbask_mean84','ntrips28','ndact28']
inc=[]
for c in pool2:
    Xc = np.hstack([Xg, Xn[:,[cols_n.index(c)]]])
    inc.append((ridge_eval(Xc,yb,db)[0],c))
inc.sort()
for mae,c in inc[:8]: print("  +%-16s %.3f (delta %+.3f)"%(c,mae,mae-g_mae))

# population-split hinges from E001
m = t.merge(E1[['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','zero_recent']], on=KEY, how='left')
s123 = m.spend_l1+m.spend_l2+m.spend_l3
act = (m.zero_recent==0).astype(float).values
inact = 1-act
H = pd.DataFrame({
 'h_act_s123': s123.values*act,
 'h_inact_s123': s123.values*inact,
 'h_act_l1': m.spend_l1.values*act,
 'h_inact_l1': m.spend_l1.values*inact,
 'h_act_pow90': np.power(1+s123.values,0.9)*act,
 'h_inact_pow90': np.power(1+s123.values,0.9)*inact,
 'h_inact_flag': inact,
})
Xh = H.values.astype(float)
for k,c in enumerate(H.columns):
    i = ridge_eval(np.hstack([Xg,Xh[:,[k]]]),yb,db)[0]
    print("  hinge %-16s incr %.3f (delta %+.3f)"%(c,i,i-g_mae))
# all hinges together
print("  all hinges      incr %.3f (delta %+.3f)"%(ridge_eval(np.hstack([Xg,Xh]),yb,db)[0], ridge_eval(np.hstack([Xg,Xh]),yb,db)[0]-g_mae))