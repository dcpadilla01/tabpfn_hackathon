import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets(); F = A.load_saved("e004_features.parquet")
d = F.merge(tt, on=["household_key","snapshot_day"])
# naive predictors
for c in ["spend_28","spend_84","spend_112"]:
    d["p_"+c] = d[c]* (28/ (28 if c=="spend_28" else 84 if c=="spend_84" else 112))
    print(c, "MAE", np.abs(d.p_ - d.future_spend_4w if False else d["p_"+c]-d.future_spend_4w).mean().round(2))
# blend naive with 0.5 weight
d["p_mix"] = 0.5*d.p_spend_28 + 0.5*d.p_spend_84
print("mix MAE", np.abs(d.p_mix-d.future_spend_4w).mean().round(2))
# residual structure of e5 on train? we don't have train preds saved. Use holdout-style: check target vs spend_28 quantiles
d["b"] = pd.qcut(d.spend_28, 10, duplicates="drop")
print(d.groupby("b", observed=True).agg(mean_t=("future_spend_4w","mean"), mean_s28=("spend_28","mean"), n=("future_spend_4w","size")).round(1))
# zero-households: what do they buy after a 0-spend 28d window?
z = d[d.spend_28<=1]
print("spend28<=1: n", len(z), "mean target", z.future_spend_4w.mean().round(1), "zerofrac", (z.future_spend_4w==0).mean().round(2))
