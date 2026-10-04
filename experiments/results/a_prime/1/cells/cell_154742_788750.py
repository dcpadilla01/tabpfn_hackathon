import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

# 1) spend_l13 validity: NaN rate by snapshot, corr among tenure>=364
print('spend_l13 NaN rate by snapshot:')
print(tr.groupby('snapshot_day').spend_l13.apply(lambda s: round(float(s.isna().mean()),3)))
sub = tr[tr.spend_l13.notna()]
print('n valid l13:', len(sub), 'corr l13 vs y:', round(float(sub.spend_l13.corr(sub.future_spend_4w)),3),
      'MAE l13 alone:', round(float((sub.spend_l13-sub.future_spend_4w).abs().mean()),2))
sub2 = sub[sub.tenure>=364]
print('tenure>=364 n:', len(sub2), 'corr:', round(float(sub2.spend_l13.corr(sub2.future_spend_4w)),3))

# 2) simple ridge on core features -> residuals; then corr of candidates with residual
feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
X = tr[feats].fillna(0).values
X = np.column_stack([np.ones(len(X)), X])
# standardize
mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
lam = 10.0
A_ = Xs.T@Xs + lam*np.eye(Xs.shape[1]); A_[0,0]-=lam
w = np.linalg.solve(A_, Xs.T@y)
pred = Xs@w
res = y - pred
print('ridge train MAE:', round(float(np.abs(res).mean()),2))

# candidates to correlate with |res| and res
cands = ['spend_l13','spend_total','disc_share84','evening_share84','stores_l1','prods_l1','prods_l2',
         'avg_basket_l2','avg_basket_l3','days_active_l2','days_active_l3','active_share_l1','spend_l456_mean',
         'spend28_GROCERY','share84_GROCERY','share84_PRODUCE','share84_MEAT','private_share84','div84']
for c in cands:
    v = tr[c].fillna(0).values
    print(c, 'corr_res', round(float(np.corrcoef(v,res)[0,1]),3), 'corr_absres', round(float(np.corrcoef(v,np.abs(res))[0,1]),3))
print('corr pred vs y:', round(float(np.corrcoef(pred,y)[0,1]),3))
print('res std:', round(float(res.std()),2), 'y std:', round(float(y.std()),2))
