
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"])
print(g.to_string())
# trend check: linear fit of mean spend vs snapshot day
d = g.reset_index()
z = np.polyfit(d.snapshot_day, d["mean"], 1)
print("slope per 28d:", z[0], "intercept:", z[1])
med = tt.groupby("snapshot_day").future_spend_4w.median().reset_index()
z2 = np.polyfit(med.snapshot_day, med.future_spend_4w, 1)
print("median slope:", z2[0], z2[1])
# OOF bias by day
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
for c in ["med_v3","hgbq_all"]:
    cv["e_"+c] = cv[c]-cv.future_spend_4w
print(cv.groupby("snapshot_day")[["e_med_v3","e_hgbq_all"]].mean().round(2).to_string())
# E016 exact OOF blend
e016_oof = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
print("E016 blend OOF MAE:", np.abs(e016_oof-cv.future_spend_4w).mean().round(4))
print("bias of E016 blend:", (e016_oof-cv.future_spend_4w).mean().round(3))
