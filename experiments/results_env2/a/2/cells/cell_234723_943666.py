import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_56","spend_84","active_28","days_since_last"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
print("spend_28:", mae(m.spend_28))
print("spend_84:", mae(m.spend_84))
print("0.5*(spend28,med):", mae(0.5*m.spend_28+0.5*m.oof_med))
print("0.7*med+0.3*spend28:", mae(0.7*m.oof_med+0.3*m.spend_28))
print("med only:", mae(m.oof_med))
# conditional median of y given spend_28 bins
bins = pd.qcut(m.spend_28, 12, duplicates="drop")
print(m.groupby(bins, observed=True).agg(y_med=("future_spend_4w","median"), y_mean=("future_spend_4w","mean"), s28=("spend_28","median"), n=("future_spend_4w","size")))
