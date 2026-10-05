
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
pop = feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean"),
                                        pop_mean_s112=("spend_112","mean"), n_hh=("household_key","count")).reset_index()
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(pop, on="snapshot_day")
print(d.round(3).to_string())
print("corr shift vs pop_mean_s28:", d["shift"].corr(d["pop_mean_s28"]).round(3))
print("corr shift vs pop_mean_s84:", d["shift"].corr(d["pop_mean_s84"]).round(3))
print("corr shift vs day:", d["shift"].corr(d["snapshot_day"]).round(3))
z = np.polyfit(d.snapshot_day, d["shift"], 1); print("shift~day fit:", z)
pred_sh = np.polyval(z, d.snapshot_day)
print("resid std:", (d["shift"]-pred_sh).std().round(2), "meanabs:", np.abs(d["shift"]-pred_sh).mean().round(2))
