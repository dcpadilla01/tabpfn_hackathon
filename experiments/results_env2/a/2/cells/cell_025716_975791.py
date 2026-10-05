
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
print(d[["snapshot_day","shift","sh_pred"]].round(2).to_string())
print("resid:", (d["shift"]-d.sh_pred).round(2).tolist())
