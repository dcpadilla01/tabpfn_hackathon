
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean")).reset_index(), on="snapshot_day")
X = np.column_stack([np.ones(len(d)), d.pop_mean_s28, d.pop_mean_s84])
b,*_ = np.linalg.lstsq(X, d["shift"].values, rcond=None)
d["sh_pred"] = X@b
# apply to validation predictions
p = agent_api.load_saved("pred_e016.parquet")
pv = p.merge(feats[["household_key","snapshot_day","spend_28","spend_84"]].rename(columns={"spend_28":"s28","spend_84":"s84"}), on=["household_key","snapshot_day"])
pv["sh"] = b[0] + b[1]*pv.s28 + b[2]*pv.s84
pv["prediction"] = (pv.prediction + pv.sh).clip(lower=0)
out = pv[["household_key","snapshot_day","prediction"]]
print(out.shape, out.snapshot_day.value_counts().to_dict())
agent_api.save_table(out, "pred_e018")
