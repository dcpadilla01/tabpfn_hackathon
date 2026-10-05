import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# median-objective model bias by quantile
for q in [0.1,0.25,0.5,0.75,0.9]:
    print("q",q, "pred_q", round(np.quantile(m.oof_med,q),1), "y_q", round(np.quantile(y,q),1))
# what if we scale the median predictions slightly?
for s in [1.0,1.05,1.1,1.15,1.2,1.25]:
    print("scale",s, mae(m.oof_med*s))
# shift (MAE-optimal shift)
for sh in [0,2,4,6,8]:
    print("shift",sh, mae(m.oof_med+sh))
