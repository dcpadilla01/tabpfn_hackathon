import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
bins = pd.qcut(m.spend_28, 10, duplicates="drop")
g = m.groupby(bins, observed=True).agg(y_med=("future_spend_4w","median"), y_mean=("future_spend_4w","mean"),
    p_sq=("oof_sq","median"), p_med=("oof_med","median"), p_log=("oof_log","median"), n=("future_spend_4w","size"))
print(g)
# MAE within each decile for each model
for c in ["oof_sq","oof_med","oof_log"]:
    e = np.abs(m[c].values-y)
    print(c, [round(e[bins.values==b].mean(),1) for b in sorted(bins.unique(), key=str)])
# zero-spend-28 households: what do models predict?
z = m[m.spend_28==0]
print("spend28==0: n", len(z), "y mean", round(z.future_spend_4w.mean(),2), "y median", z.future_spend_4w.median(),
      "pred med median", z.oof_med.median(), "MAE med", round(np.abs(z.oof_med-z.future_spend_4w).mean(),2))
