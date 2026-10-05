
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
def mae(df, col):
    m = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    return m.dropna(subset=[col]).groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g[col]).mean())
cv = agent_api.load_saved("oof_e016_cv.parquet")
for c in ["med_v3","med_all","hgbq_v3","hgbq_all"]:
    m = tt.merge(cv[["household_key","snapshot_day",c]], on=["household_key","snapshot_day"])
    print(c, "MAE", np.abs(m[c]-m.future_spend_4w).mean().round(3), "n", m[c].notna().sum())
    print(m.dropna(subset=[c]).groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g[c]).mean()).round(2).to_dict())
# blend weights search on OOF
cv = cv.merge(tt, on=["household_key","snapshot_day"]).dropna()
y = cv.future_spend_4w.values
best=None
for wm in np.linspace(0,1,21):
    for wv in np.linspace(0,1,21):
        p = wm*cv.med_v3.values + wv*cv.hgbq_v3.values + (1-wm-wv)*cv.med_all.values
        if (1-wm-wv)<-0.001: continue
        e = np.abs(p-y).mean()
        if best is None or e<best[0]: best=(e,wm,wv)
print("best OOF blend med_v3/hgbq_v3/med_all:", best)
for wm in np.linspace(0,1,21):
    for wv in np.linspace(0,1,21):
        p = wm*cv.med_v3.values + wv*cv.hgbq_all.values + (1-wm-wv)*cv.med_all.values
        if (1-wm-wv)<-0.001: continue
        e = np.abs(p-y).mean()
        if best is None or e<best[0]: best=(e,wm,wv,"v3/all")
print("best overall:", best)
