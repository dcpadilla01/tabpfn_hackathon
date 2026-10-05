import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 200)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
bins = pd.qcut(m.spend_28, 10, duplicates="drop")
g = m.groupby(bins, observed=True).agg(y_med=("future_spend_4w","median"), y_mean=("future_spend_4w","mean"),
    p_sq=("oof_sq","median"), p_med=("oof_med","median"), p_log=("oof_log","median"), n=("future_spend_4w","size"))
print(g.to_string())
# high-spend households: bias
hi = m[m.spend_28>300]
print("\nspend28>300: n", len(hi), "y mean", round(hi.future_spend_4w.mean(),1), "y med", hi.future_spend_4w.median(),
      "| pred sq", round(hi.oof_sq.median(),1), "med", round(hi.oof_med.median(),1), "log", round(hi.oof_log.median(),1))
print("MAE hi: sq", round(np.abs(hi.oof_sq-hi.future_spend_4w).mean(),1), "med", round(np.abs(hi.oof_med-hi.future_spend_4w).mean(),1), "log", round(np.abs(hi.oof_log-hi.future_spend_4w).mean(),1))
# how much of total error from top decile of spend_28?
abs_err = np.abs(m.oof_med.values - y)
print("\nerr share by spend28 decile (med model):")
for b in sorted(bins.unique(), key=str):
    mask = (bins.values==b)
    print(str(b)[:25], round(abs_err[mask].sum()/abs_err.sum(),3))
