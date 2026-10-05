
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean"),
        pop_mean_s112=("spend_112","mean"), pop_mean_s224=("spend_224","mean"),
        pop_mean_s56=("spend_56","mean"), pop_mean_b28=("baskets_28","mean")).reset_index(), on="snapshot_day")
# regression: shift ~ a + b*pop_mean_s28 + c*pop_mean_s84 (+ day)
X = np.column_stack([np.ones(len(d)), d.pop_mean_s28, d.pop_mean_s84, d.snapshot_day])
for cols,label in [([0,1],"s28"),([0,1,2],"s28+s84"),([0,1,2,3],"+day"),([0,3],"day only"),([0,2],"s84 only")]:
    b, *_ = np.linalg.lstsq(X[:,cols], d["shift"].values, rcond=None)
    r = d["shift"].values - X[:,cols]@b
    print(label, "coef", b.round(4), "resid std", r.std().round(2), "meanabs", np.abs(r).mean().round(2))
