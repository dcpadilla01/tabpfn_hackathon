import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84","active_112","baskets_112"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# gap_mean_112: median gap between baskets over 112d
print("corr spend_28 vs y:", round(np.corrcoef(m.spend_28, y)[0,1],3))
# households with long gaps (infrequent): how do they behave?
for lo,hi in [(0,7),(7,14),(14,28),(28,60),(60,10**9)]:
    s = m[(m.gap_mean_112>=lo)&(m.gap_mean_112<hi)&(m.spend_28>0)]
    print(f"gap[{lo},{hi}) n={len(s)} y_med={s.future_spend_4w.median():.0f} y_mean={s.future_spend_4w.mean():.0f} s28_med={s.spend_28.median():.0f} p_med={s.oof_med.median():.0f} MAE_med={np.abs(s.oof_med-s.future_spend_4w).mean():.1f}")
# days_since_last effect
for lo,hi in [(0,7),(7,14),(14,28),(28,56),(56,10**9)]:
    s = m[(m.days_since_last>=lo)&(m.days_since_last<hi)]
    print(f"dsl[{lo},{hi}) n={len(s)} y_med={s.future_spend_4w.median():.0f} p_med={s.oof_med.median():.0f} MAE={np.abs(s.oof_med-s.future_spend_4w).mean():.1f}")
