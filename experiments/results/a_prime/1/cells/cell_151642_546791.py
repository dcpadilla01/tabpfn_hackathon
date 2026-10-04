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
Xn,yn,dn,cols_n = prep(NF)
base = ridge_eval(Xb,yb,db)[0]
print("base %.3f"%base)

# (a) greedy forward selection of candidate additions
chosen=[]; cur=Xb; cur_mae=base
pool = ['nwmax12','nbask_max84','nwstd12','nspend_l12','nspend7','nf_ncv12','nf_nspend28_pow90','nunits84','nprods84','nf_pow90_ewm4','ngap_std','nspend_l11']
for it in range(6):
    best=None
    for c in pool:
        if c in chosen: continue
        j = cols_n.index(c)
        Xc = np.hstack([cur, Xn[:,[j]]])
        mae,_ = ridge_eval(Xc,yb,db)
        if best is None or mae<best[0]: best=(mae,c,j)
    if best[0] < cur_mae - 1e-4:
        cur_mae, c, j = best; chosen.append(c); cur = np.hstack([cur, Xn[:,[j]]])
        print("greedy +%-18s -> %.3f"%(c,cur_mae))
    else:
        print("no improvement; stop"); break
print("chosen:", chosen)

# (b) pruning: drop weakest E003 features by univariate
uni=[]
for j,c in enumerate(cols_b):
    mae,_=ridge_eval(Xb[:,[j]],yb,db); uni.append((mae,c,j))
uni.sort()
weak = [c for _,c,_ in uni[-15:]]
keep_idx = [j for _,_,j in uni if _ not in [u[0] for u in uni[-15:]]]
Xp = Xb[:, [j for _,_,j in uni[:-15]]]
print("pruned-15 offline: %.3f"%ridge_eval(Xp,yb,db)[0])
Xp2 = Xb[:, [j for _,_,j in uni[:-8]]]
print("pruned-8  offline: %.3f"%ridge_eval(Xp2,yb,db)[0])

# (c) zero/intermittency composites built from E001 raw windows
E1 = api.load_saved('e001_txhist.parquet')
m = t.merge(E1[['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','spend_l13','zero_recent','spend_l123_mean','spend_l456_mean','momentum']], on=KEY, how='left')
zz = {}
zz['n_zerowin_l6'] = ((m[['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6']]<=0).sum(axis=1)).astype(float)
zz['n_zerowin_l3'] = ((m[['spend_l1','spend_l2','spend_l3']]<=0).sum(axis=1)).astype(float)
zz['nf_blend'] = 0.5*m.spend_l123_mean + 0.3*m.spend_l456_mean + 0.2*m.spend_l13/13*13
zz['nf_blend2'] = 0.6*m.spend_l123_mean + 0.4*m.spend_l456_mean
zz['nf_min_l3'] = m[['spend_l1','spend_l2','spend_l3']].min(axis=1)
zz['nf_max_l3'] = m[['spend_l1','spend_l2','spend_l3']].max(axis=1)
zz['nf_std_l3'] = m[['spend_l1','spend_l2','spend_l3']].std(axis=1)
zz['nf_zerowin_x_spend'] = zz['n_zerowin_l6']*m.spend_l123_mean
zz['nf_mom_x_spend'] = m.momentum*m.spend_l123_mean
zz['nf_pow90_blend'] = np.power(1+zz['nf_blend'],0.9)
Z = pd.DataFrame(zz)
Xz = Z.values.astype(float)
print("\nZero/intermittency composites (univariate + incremental over E003):")
for k,c in enumerate(zz):
    u = ridge_eval(Xz[:,[k]],yb,db)[0]
    i = ridge_eval(np.hstack([Xb,Xz[:,[k]]]),yb,db)[0]
    print("  %-22s uni %7.2f  incr %7.3f (delta %+.3f)"%(c,u,i,i-base))