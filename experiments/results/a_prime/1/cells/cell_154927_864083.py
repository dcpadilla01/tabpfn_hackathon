import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
Xall = tr[feats].fillna(0).values.astype(float)

def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    w = np.linalg.solve(Am, Xs.T@yy)
    return w, mu, sd

def apply_w(w, mu, sd, X):
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    return Xs@w

w, mu, sd = fit_w(Xall, y)
pred_all = apply_w(w, mu, sd, Xall)
print('global ridge train MAE:', round(float(np.abs(pred_all-y).mean()),2))

snaps = sorted(tr.snapshot_day.unique())
resid_by_snap = {}
for s in snaps:
    m = (tr.snapshot_day==s).values
    w2, mu2, sd2 = fit_w(Xall[~m], y[~m])
    p = apply_w(w2, mu2, sd2, Xall[m])
    resid_by_snap[s] = float((y[m]-p).mean())
print('per-snapshot mean residual (LOSO):', {k: round(v,1) for k,v in resid_by_snap.items()})
print('spread:', round(max(resid_by_snap.values())-min(resid_by_snap.values()),1))

res = y - pred_all
print('corr resid vs snapshot_day:', round(float(np.corrcoef(tr.snapshot_day, res)[0,1]),3))
cap = np.percentile(y, 99)
print('y capped at 99th pct -> MAE:', round(float(np.abs(np.minimum(pred_all, cap)-y).mean()),2))
