import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e010_l13fix.parquet')
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
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w
w, mu, sd = fit_w(Xall, y)
pred = apply_w(w, mu, sd, Xall)
res = y - pred

# residual corr among REAL l13 rows
m = tr.has_real_l13==1
print('corr resid vs l13 (real rows):', round(float(np.corrcoef(tr.spend_l13[m], res[m.values])[0,1]),3))
print('corr resid vs l13_over_recent (real):', round(float(np.corrcoef(tr.l13_over_recent[m].clip(-5,5), res[m.values])[0,1]),3))
print('corr resid vs has_real_l13:', round(float(np.corrcoef(tr.has_real_l13, res)[0,1]),3))

# what fraction of val rows would be affected?
val = df[df.future_spend_4w.isna()]
print('val rows with real l13:', int((val.has_real_l13==1).sum()), '/', len(val))
# distribution shift check: spend_l123_mean train vs val
print('train l123 mean:', round(float(tr.spend_l123_mean.mean()),1), 'val:', round(float(val.spend_l123_mean.mean()),1))
print('train tenure mean:', round(float(tr.tenure.mean()),1), 'val:', round(float(val.tenure.mean()),1))
# key: among val rows, how many have tenure < 364 (i.e., would have had fake l13 in E003)?
print('val rows tenure<364:', int((val.tenure<364).sum()), 'of', len(val))
# check l13 vs y relationship stability across snapshots among real rows
for s in sorted(sub.snapshot_day.unique()):
    ss = sub[sub.snapshot_day==s]
    print(s, 'n', len(ss), 'corr', round(float(ss.spend_l13.corr(ss.future_spend_4w)),3), 'mean y', round(float(ss.future_spend_4w.mean()),1))
