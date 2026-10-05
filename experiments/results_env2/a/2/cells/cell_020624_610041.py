import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
oof = A.load_saved("oof_e008.parquet")
m = oof.merge(tt, on=["household_key","snapshot_day"])
m["res_sq"] = m.future_spend_4w - m.oof_sq
m["res_med"] = m.future_spend_4w - m.oof_med
m["res_log"] = m.future_spend_4w - m.oof_log
g = m.groupby("snapshot_day")[["res_sq","res_med","res_log"]].agg(["median","mean"])
print("OOF residuals (y - pred) by snapshot day:")
print(g.round(2).to_string())
print("\nOOF MAE by day:")
print(m.groupby("snapshot_day")[["res_sq","res_med","res_log"]].apply(lambda d: d.abs().mean()).round(2).to_string())
print("\nOverall median residuals:", m[["res_sq","res_med","res_log"]].median().round(2).to_dict())
print("Overall mean residuals:", m[["res_sq","res_med","res_log"]].mean().round(2).to_dict())
# what shift minimizes OOF MAE for each model?
for c in ["res_sq","res_med","res_log"]:
    r = m[c].values
    best = min(np.arange(-15,15.25,0.25), key=lambda s: np.abs(r-s).mean())
    print(c, "MAE", round(np.abs(r).mean(),3), "-> best shift", best, "MAE after", round(np.abs(r-best).mean(),3))
