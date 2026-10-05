
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
# per-snapshot mean of household-level 'median-ish' features vs realized mean target
d = feats.groupby("snapshot_day").agg(pop_med_s28=("spend_28","median"), pop_mean_s28=("spend_28","mean"),
                                      pop_mean_s84=("spend_84","mean"), pop_mean_s112=("spend_112","mean")).reset_index()
d = d.merge(tt.groupby("snapshot_day").future_spend_4w.agg(t_mean="mean", t_med="median").reset_index(), on="snapshot_day")
print(d.round(2).to_string())
print("corr t_mean vs pop_mean_s28:", d.t_mean.corr(d.pop_mean_s28).round(3))
print("corr t_mean vs pop_mean_s84:", d.t_mean.corr(d.pop_mean_s84).round(3))
print("corr t_mean vs pop_mean_s112:", d.t_mean.corr(d.pop_mean_s112).round(3))
# ratio t_mean / pop_mean_s28
print("ratio t_mean/pop_mean_s28:", (d.t_mean/d.pop_mean_s28).round(3).tolist())
