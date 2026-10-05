import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e011, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e011.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)

def ridge_variants(X, y, days, val_day=431):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    A = np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); B = np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
    out={}
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
        out[f'ridge raw lam{lam}'] = float(np.mean(np.abs(B@w - y[te])))
    ly = np.log1p(y)
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@ly[tr])
        out[f'ridge log1p lam{lam}'] = float(np.mean(np.abs(np.expm1(B@w) - y[te])))
    # robust: huber via IRLS on raw
    w = np.zeros(X.shape[1]); 
    for it in range(15):
        r = y[tr]-A@w; s = np.median(np.abs(r-np.median(r)))*4+1e-9
        wt = np.minimum(1.0, s/np.maximum(np.abs(r),1e-9))
        W = A*(wt[:,None]); w = np.linalg.solve(W.T@A+30*np.eye(X.shape[1]), W.T@y[tr])
    out['huber raw'] = float(np.mean(np.abs(B@w - y[te])))
    ly = np.log1p(y)
    w = np.zeros(X.shape[1])
    for it in range(15):
        r = ly[tr]-A@w; s = np.median(np.abs(r-np.median(r)))*4+1e-9
        wt = np.minimum(1.0, s/np.maximum(np.abs(r),1e-9))
        W = A*(wt[:,None]); w = np.linalg.solve(W.T@A+30*np.eye(X.shape[1]), W.T@ly[tr])
    out['huber log1p'] = float(np.mean(np.abs(np.expm1(B@w) - y[te])))
    # knn on standardized features (cosine/cityblock)
    from collections import Counter
    best=[]
    for k in [15,40]:
        # subsample train for speed
        idx_tr = np.where(tr)[0]
        D = np.abs(A[idx_tr][:,None,:]-B[None,:,:]).sum(-1) if False else None
        # chunked L1
        n_te = te.sum(); preds=np.zeros(n_te)
        CH=500
        for c0 in range(0,n_te,CH):
            sl = slice(c0,min(c0+CH,n_te))
            dist = np.abs(A[idx_tr][:,None,:]-B[sl][None,:,:]).sum(-1)
            nn = np.argpartition(dist, k, axis=0)[:k]
            preds[sl] = y[tr][nn].mean(0)
        out[f'knn L1 k={k}'] = float(np.mean(np.abs(preds-y[te])))
    return out

t0=time.time()
for k,v in ridge_variants(X,y,days).items(): print(f'{k:24s} {v:8.3f}')
print(f'{time.time()-t0:.0f}s')
