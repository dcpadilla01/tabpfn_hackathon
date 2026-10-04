import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
cats = [c for c in t.columns if str(t[c].dtype)=='category' or t[c].dtype==bool]
print('categorical/bool cols:', cats)
for c in cats: print(c, t[c].value_counts(dropna=False).head(3).to_dict())

feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
print('n feat cols', len(feat_cols))

def ridge_eval(X, y, tr_mask, va_mask, alphas=(30,100,300,1000,3000)):
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0)+1e-9
    Xs = (X-mu)/sd
    Xtr, ytr = Xs[tr_mask], y[tr_mask]
    G = Xtr.T@Xtr; b = Xtr.T@ytr
    best=None
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(G.shape[0]), b)
        pred = Xs[va_mask]@w
        mae = np.abs(pred-y[va_mask]).mean()
        if best is None or mae<best[0]: best=(mae,a,w)
    return best

train_snaps = sorted(tt.snapshot_day.unique())
print('train snaps', train_snaps)
m_tr = df.snapshot_day<=403; m_va = df.snapshot_day==431
X = df[feat_cols].copy()
for c in cats:
    X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
X = X.astype(np.float64).values
y = df.future_spend_4w.values
mae,a,w = ridge_eval(X, y, m_tr.values, m_va.values)
print('internal holdout snap431 MAE %.3f alpha %d'%(mae,a))
m_tr2 = df.snapshot_day!=403; m_va2 = df.snapshot_day==403
mae2,a2,_ = ridge_eval(X, y, m_tr2.values, m_va2.values)
print('internal holdout snap403 MAE %.3f alpha %d'%(mae2,a2))
for col in ['spend_28','tlag_mean','ewma_4']:
    if col in df.columns:
        print(col, 'MAE431 %.2f'%np.abs(df.loc[m_va,col].values-y[m_va.values]).mean())
