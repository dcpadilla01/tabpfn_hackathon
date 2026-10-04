
import agent_api as A
import pandas as pd, numpy as np

# pseudo-validation at snapshot 431: how much headroom do simple predictors have?
u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
tr = m[m.snapshot_day <= 403]; te = m[m.snapshot_day == 431]
ytr, yte = tr['future_spend_4w'].values, te['future_spend_4w'].values

def mae(p, y): return np.mean(np.abs(p - y))

# baseline: global median
print("const median MAE@431:", round(mae(np.median(ytr), yte),2))
# spend_l1 raw
print("spend_l1 MAE@431:", round(mae(te['spend_l1'].values, yte),2))
# scaled spend_l1: fit a,b on train via least squares on (spend_l1 -> target)
X = tr['spend_l1'].values; 
b, a = np.polyfit(X, ytr, 1)
print("lin(spend_l1) MAE@431:", round(mae(a + b*te['spend_l1'].values, yte),2))
# log-log
Xl = np.log1p(tr['spend_l1'].values); yl = np.log1p(ytr)
b, a = np.polyfit(Xl, yl, 1)
p = np.expm1(a + b*np.log1p(te['spend_l1'].values))
print("loglog(spend_l1) MAE@431:", round(mae(p, yte),2))
# bin median of spend_l1 (20 bins) fit on train
bins = np.quantile(tr['spend_l1'], np.linspace(0,1,21))
bins[0]=-1e9; bins[-1]=1e9
bid = np.digitize(tr['spend_l1'], bins) 
bmed = pd.Series(ytr).groupby(bid).median()
p = np.array([bmed.get(np.digitize(v,bins), np.median(ytr)) for v in te['spend_l1'].values])
print("binned-median(spend_l1) MAE@431:", round(mae(p, yte),2))
# two-feature binned: spend_l1 x spend_l4 ratio
print()
print("target stats train: mean %.1f med %.1f | @431: mean %.1f med %.1f" % (ytr.mean(), np.median(ytr), yte.mean(), np.median(yte)))
