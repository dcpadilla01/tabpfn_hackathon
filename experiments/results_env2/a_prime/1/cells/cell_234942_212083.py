
import numpy as np, pandas as pd

t = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
key = ['household_key','snapshot_day']
df = t.merge(tt, on=key, how='left')
feat_cols = [c for c in t.columns if c not in key]
X = df[feat_cols].copy()
cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype('category').cat.codes.astype(float).replace(-1.0, np.nan)
Xv = X.values.astype(float)
is_tr = df['future_spend_4w'].notna().values
y_raw = df.loc[is_tr, 'future_spend_4w'].values.astype(float)
yt = np.log1p(y_raw)
Xtr = Xv[is_tr]
mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[sd==0]=1.0
Xs = np.nan_to_num((Xv - mu)/sd, nan=0.0, posinf=0.0, neginf=0.0)
Xt = Xs[is_tr]
ymu = yt.mean(); ysd = yt.std(); yc = (yt - ymu)/ysd
days_all = df['snapshot_day'].values.astype(int)
days_tr = days_all[is_tr]
tr_days = sorted(set(days_tr))

def fit_pred(Xa, ya, Xb, alpha):
    return Xb @ np.linalg.solve(Xa.T @ Xa + alpha*np.eye(Xa.shape[1]), Xa.T @ ya)

grid = [3,10,30,100,300,1000,3000]
best = None
for a in grid:
    pred = np.zeros(len(yt))
    for s in tr_days:
        m = days_tr==s
        pred[m] = fit_pred(Xt[~m], yc[~m], Xt[m], a)
    pl = np.clip(pred*ysd + ymu, 0, 7)
    mae = np.mean(np.abs(np.expm1(pl) - y_raw))
    print('alpha', a, 'oof MAE', round(mae,3))
    if best is None or mae < best[1]: best = (a, mae, pred.copy())
a, omae, oof = best
print('best alpha', a, 'oof MAE', round(omae,3))
w = fit_pred(Xt, yc, Xs, a)
full_pred = np.clip(w*ysd + ymu, 0, 7)
stack_log = full_pred.copy()
stack_log[is_tr] = np.clip(oof*ysd+ymu, 0, 7)
out = t.copy()
out['stack_ridge_log'] = stack_log
out['stack_ridge'] = np.expm1(stack_log)
p = agent_api.save_table(out, 'e015_stack.parquet')
print('saved', p, out.shape)
