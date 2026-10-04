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
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w
w, mu, sd = fit_w(Xall, y)
pred = apply_w(w, mu, sd, Xall)

zero = y==0
print('y=0 rows:', zero.sum(), 'mean pred on them:', round(float(pred[zero].mean()),1),
      'MAE contribution:', round(float(np.abs(pred[zero]).mean()*zero.mean()),2))
print('y>0 rows: mean |err|:', round(float(np.abs((pred-y))[~zero]).mean(),1) if False else round(float(np.abs(pred[~zero]-y[~zero]).mean()),2))

# how separable are y=0 households?
print('\nAmong y=0: spend_l1 quantiles:', np.percentile(tr.spend_l1[zero], [10,25,50,75,90]).round(1))
print('Among y>0: spend_l1 quantiles:', np.percentile(tr.spend_l1[~zero], [10,25,50,75,90]).round(1))
print('spend_l1==0 -> P(y=0):', round(float((y[tr.spend_l1==0]==0).mean()),3), 'n=', int((tr.spend_l1==0).sum()))
print('spend_l1>0 -> P(y=0):', round(float((y[tr.spend_l1>0]==0).mean()),3))
print('days_since_last>28 -> P(y=0):', round(float((y[tr.days_since_last>28]==0).mean()),3), 'n=', int((tr.days_since_last>28).sum()))
print('days_since_last<=28 -> P(y=0):', round(float((y[tr.days_since_last<=28]==0).mean()),3))

# error decomposition: how much would perfect zero-classification help?
# if we could perfectly identify y=0 and set pred=0 there:
pred2 = pred.copy(); pred2[zero]=0
print('\nMAE now:', round(float(np.abs(pred-y).mean()),2), 'MAE with oracle zero:', round(float(np.abs(pred2-y).mean()),2))
# if we just clipped predictions at 0..cap: no change. What about shrink of positives?
# contribution of y=0 rows to MAE:
print('MAE on y=0 rows:', round(float(np.abs(pred[zero]).mean()),2), 'weight', round(float(zero.mean()),3))
print('MAE on y>0 rows:', round(float(np.abs(pred[~zero]-y[~zero]).mean()),2))
