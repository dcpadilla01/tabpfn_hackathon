import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e010_l13fix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
sub = tr[tr.has_real_l13==1]
for s in sorted(sub.snapshot_day.unique()):
    ss = sub[sub.snapshot_day==s]
    print(s, 'n', len(ss), 'corr', round(float(ss.spend_l13.corr(ss.future_spend_4w)),3), 'mean y', round(float(ss.future_spend_4w.mean()),1))
# l13 vs l123_mean among real rows: is l13 just a noisy duplicate?
print('\nreal rows: corr l13 vs l123_mean:', round(float(sub.spend_l13.corr(sub.spend_l123_mean)),3))
# per-household: does l13 add info beyond l1..l6? quick ridge with/without
feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w
Xa = sub[feats].fillna(0).values.astype(float); ya = sub.future_spend_4w.values
w0,mu0,sd0 = fit_w(Xa, ya); p0 = apply_w(w0,mu0,sd0,Xa)
Xb = np.column_stack([Xa, sub.spend_l13.fillna(0).values]); 
w1,mu1,sd1 = fit_w(Xb, ya); p1 = apply_w(w1,mu1,sd1,Xb)
print('train MAE without l13:', round(float(np.abs(p0-ya).mean()),2), 'with l13:', round(float(np.abs(p1-ya).mean()),2))
# LOSO-style: hold out last snapshot among real rows
smax = sub.snapshot_day.max()
m = sub.snapshot_day==smax
w2,mu2,sd2 = fit_w(Xa[~m.values], ya[~m.values]); p2 = apply_w(w2,mu2,sd2,Xa[m.values])
w3,mu3,sd3 = fit_w(np.column_stack([Xa, sub.spend_l13.fillna(0).values])[~m.values], ya[~m.values]); p3 = apply_w(w3,mu3,sd3,np.column_stack([Xa, sub.spend_l13.fillna(0).values])[m.values])
print('holdout(431) MAE without l13:', round(float(np.abs(p2-ya[m.values]).mean()),2), 'with l13:', round(float(np.abs(p3-ya[m.values]).mean()),2))
