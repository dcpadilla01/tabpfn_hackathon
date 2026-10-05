import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
# check target autocorrelation: y vs lag364_spend (year-ago same window) on train rows
fs = A.load_saved("feats_seasonal.parquet")
m = fs.merge(tt, on=["household_key","snapshot_day"])
m = m[m.snapshot_day>=263]  # where lag364 exists
print("n:", len(m))
print("corr(lag364_spend, y):", np.corrcoef(m.lag364_spend, m.future_spend_4w)[0,1].round(3))
print("corr(spend_28, y):", np.corrcoef(m.spend_28 if 'spend_28' in m else m.lag84_spend, m.future_spend_4w)[0,1].round(3))
# blend check: y_hat = 0.5*lag364 + 0.5*recent28
m["blend"] = 0.5*m.lag364_spend + 0.5*m.lag84_spend
for c in ["lag364_spend","lag84_spend","blend"]:
    print(c, "MAE:", round(np.abs(m[c]-m.future_spend_4w).mean(),2))
# what about combining recent28? need feats_v3
f3 = A.load_saved("feats_v3.parquet")
m2 = m.merge(f3[["household_key","snapshot_day","spend_28","spend_84"]], on=["household_key","snapshot_day"])
for w in [0.0,0.25,0.5,0.75,1.0]:
    b = w*m2.lag364_spend + (1-w)*m2.spend_28
    print(f"blend w={w} (lag364 vs spend28) MAE:", round(np.abs(b-m2.future_spend_4w).mean(),2))
