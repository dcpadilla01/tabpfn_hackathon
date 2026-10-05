import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# per-decile optimal scaling of med predictions (to see if heteroscedastic bias)
bins = pd.qcut(m.spend_28, 8, duplicates="drop")
for b in sorted(bins.unique(), key=str):
    s = m[bins.values==b]
    best=(None,1e9)
    for sc in np.arange(0.7,1.5,0.05):
        e = np.mean(np.abs(s.oof_med.values*sc - s.future_spend_4w.values))
        if e<best[1]: best=(round(sc,2),round(e,2))
    print(str(b)[:28], "n",len(s), "y_mean", round(s.future_spend_4w.mean(),1), "p_med", round(s.oof_med.median(),1), "best scale", best)
