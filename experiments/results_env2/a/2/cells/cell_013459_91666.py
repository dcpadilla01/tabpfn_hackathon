import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
print(oof.snapshot_day.value_counts().sort_index())
tt = agent_api.train_targets()
m = oof.merge(tt, on=["household_key","snapshot_day"])
print(m.shape)
for c in ["oof_sq","oof_med","oof_log"]:
    print(c, "MAE", np.abs(m[c]-m.future_spend_4w).mean())
# per-snapshot-day bias
m["resid_med"] = m.future_spend_4w - m.oof_med
print(m.groupby("snapshot_day").agg(n=("resid_med","size"), mae=("resid_med", lambda s: s.abs().mean()), bias=("resid_med","mean"), tgt=("future_spend_4w","mean")))
# correlations among saved val predictions
preds = {n: agent_api.load_saved(n+".parquet") for n in ["pred_seasonal","pred_e008","pred_e009","pred_e010_blend","pred_log1p"]}
base = preds["pred_seasonal"][["household_key","snapshot_day"]].copy()
for n,p in preds.items():
    base = base.merge(p.rename(columns={"prediction":n}), on=["household_key","snapshot_day"])
print(base.shape)
print(base.drop(columns=["household_key","snapshot_day"]).corr().round(4))
print(base.drop(columns=["household_key","snapshot_day"]).describe().round(2))
