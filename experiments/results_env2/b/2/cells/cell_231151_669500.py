import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e011, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e011.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)

def variants(X, y, days, val_day=431):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    Z = np.where(np.isnan(X),0,(X-mu)/sd)
    A, B = Z[tr], Z[te]
    out={}
    ly = np.log1p(y)
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
        out[f'ridge raw lam{lam}'] = float(np.mean(np.abs(B@w - y[te])))
        w2 = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@ly[tr])
        out[f'ridge log1p lam{lam}'] = float(np.mean(np.abs(np.expm1(B@w2) - y[te])))
    # knn (indices into full Z)
    idx_tr = np.where(tr)[0]
    for k in [15,40]:
        n_te = int(te.sum()); preds=np.zeros(n_te); CH=400
        for c0 in range(0,n_te,CH):
            sl = slice(c0,min(c0+CH,n_te))
            dist = np.abs(Z[idx_tr][:,None,:]-Z[te][sl][None,:,:]).sum(-1)
            nn = np.argpartition(dist, k, axis=0)[:k]
            preds[sl] = y[tr][nn].mean(0)
        out[f'knn L1 k={k}'] = float(np.mean(np.abs(preds-y[te])))
    return out

t0=time.time()
res = variants(X,y,days)
for k,v in res.items(): print(f'{k:24s} {v:8.3f}')
print(f'{time.time()-t0:.0f}s')
