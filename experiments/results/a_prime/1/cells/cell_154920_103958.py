import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

# 1) simple per-snapshot ridge on E003 core features: does snapshot-level recalibration help?
feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
# leave-one-snapshot-out CV within train for per-snapshot calibration
snaps = sorted(tr.snapshot_day.unique())
def fit_pred(train_mask, Xall, yall):
    X = Xall[train_mask]; yy = yall[train_mask]
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    lam=10.0
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    w = np.linalg.solve(Am, Xs.T@yy)
    Xt = Xall[~train_mask]
    Xts = np.column_stack([np.ones(len(Xt)), (Xt[:,1:]-mu)/sd])
    return Xts@w, Xs@w

Xall = tr[feats].fillna(0).values.astype(float)
yall = y
# global fit
pred_all, _ = fit_pred(np.ones(len(tr),bool), Xall, yall)
print('global ridge train MAE:', round(float(np.abs(pred_all-yall).mean()),2))

# per-snapshot intercept-only recalibration via LOSO
resid_by_snap = {}
for s in snaps:
    m = tr.snapshot_day==s
    p_in, p_all = None, None
    # fit on all except s
    mask = ~m.values
    w_pred, w_in = fit_pred(mask, Xall, yall)
    resid_by_snap[s] = float((yall[m.values]-w_pred).mean())
print('per-snapshot mean residual (LOSO):')
print({k: round(v,1) for k,v in resid_by_snap.items()})
print('spread:', round(max(resid_by_snap.values())-min(resid_by_snap.values()),1))

# 2) does adding snapshot_day (linear) change anything? quick check: corr of resid with snapshot
res = yall - pred_all
print('corr resid vs snapshot_day:', round(float(np.corrcoef(tr.snapshot_day, res)[0,1]),3))

# 3) winsorize y at 99th pct of train -> effect on achievable MAE (sanity, not usable directly)
cap = np.percentile(y, 99)
print('y capped at 99th:', round(float(np.abs(np.minimum(pred_all, cap)-yall).mean()),2))
