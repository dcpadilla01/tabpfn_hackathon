import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3, on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# How well does the model predict ZERO vs nonzero (classification)?
z = (y==0)
from sklearn.metrics import roc_auc_score
print("AUC spend28 -> y>0:", round(roc_auc_score(1-z, m.spend_28),3))
print("AUC pred_med -> y>0:", round(roc_auc_score(1-z, m.oof_med),3))
print("AUC dsl -> y==0:", round(roc_auc_score(z, m.days_since_last),3))
# conditional P(y==0) by spend_28 bins
bins = pd.qcut(m.spend_28, 10, duplicates="drop")
print(m.groupby(bins, observed=True)["future_spend_4w"].agg([("p0", lambda s:(s==0).mean()), ("med", "median"), ("n","size")]))
