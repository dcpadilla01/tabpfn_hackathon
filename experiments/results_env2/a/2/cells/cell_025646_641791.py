
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
# per-snapshot optimal constant shift
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
g = cv.groupby("snapshot_day").apply(lambda x: pd.Series({
    "shift": (x.future_spend_4w-x.blend).mean(),
    "scale": (x.future_spend_4w*x.blend).sum()/(x.blend*x.blend).sum(),
    "mae0": np.abs(x.blend-x.future_spend_4w).mean()}))
print(g.round(3).to_string())
# global scale on OOF
for s in [1.0,1.05,1.1,1.15,1.2,1.25]:
    print(s, np.abs(s*cv.blend-cv.future_spend_4w).mean().round(3))
