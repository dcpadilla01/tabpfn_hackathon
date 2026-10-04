import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
cats = [c for c in feat_cols if str(t[c].dtype)=='category' or t[c].dtype==bool]

def prep(df):
    X = df[feat_cols].copy()
    for c in cats:
        X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(np.float64)
    return X

def ridge_eval(X, y, tr_idx, va_idx, alphas=(30,100,300,1000,3000)):
    Xtr_raw = X[tr_idx]
    mu = np.nanmean(Xtr_raw,0); 
    Xf = np.where(np.isnan(X), 0.0, X-mu)   # fill NaN with mean -> 0 after centering
    sd = Xf[tr_idx].std(0)+1e-9
    Xs = Xf/sd
    Xtr, ytr = Xs[tr_idx], y[tr_idx]
    G = Xtr.T@Xtr; b = Xtr.T@ytr
    best=None
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(G.shape[0]), b)
        mae = np.abs(Xs[va_idx]@w - y[va_idx]).mean()
        if best is None or mae<best[0]: best=(mae,a)
    return best

X = prep(df).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int)
m_va = (d==431).values; m_tr = (d<=403).values
print('holdout 431: MAE %.3f alpha %d'%ridge_eval(X,y,m_tr,m_va))
m_va2=(d==403).values; m_tr2=(d<=375).values
print('holdout 403: MAE %.3f alpha %d'%ridge_eval(X,y,m_tr2,m_va2))
m_va3=(d==375).values; m_tr3=(d<=347).values
print('holdout 375: MAE %.3f alpha %d'%ridge_eval(X,y,m_tr3,m_va3))
# log1p target ridge
ylog = np.log1p(y)
mae,a = ridge_eval(X,ylog,m_tr,m_va)
pred_log = None
mu = np.nanmean(X[m_tr],0); Xf=np.where(np.isnan(X),0,X-mu); sd=Xf[m_tr].std(0)+1e-9; Xs=Xf/sd
G=Xs[m_tr].T@Xs[m_tr]; b=Xs[m_tr].T@ylog[m_tr]
w=np.linalg.solve(G+a*np.eye(G.shape[0]),b)
pl = np.expm1(Xs[m_va]@w)
print('log-target holdout 431: MAE %.3f alpha %d'%(np.abs(pl-y[m_va]).mean(),a))
