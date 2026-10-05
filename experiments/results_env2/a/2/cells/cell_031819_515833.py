import agent_api as A
import pandas as pd, numpy as np, itertools

oof = A.load_saved("oof_e016_cv.parquet")
tt = A.train_targets()
oof = oof.merge(tt.rename(columns={"future_spend_4w":"y_tt"}), on=["household_key","snapshot_day"], how="left")
print("y vs y_tt mismatch:", (oof.y.fillna(-9)!=oof.y_tt.fillna(-9)).sum())

tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
y = oof.y.values
print("oof_tw MAE:", np.mean(np.abs(oof.oof_tw.values - y)).round(3))

# correlations among val preds
preds = {}
for nm in ["pred_e006","pred_e008","pred_e009","pred_e010","pred_e010_blend","pred_e014","pred_e016","pred_log1p","pred_prof","pred_pt","pred_seasonal","pred_e018"]:
    preds[nm] = A.load_saved(nm+".parquet").set_index(["household_key","snapshot_day"])["prediction"]
P = pd.DataFrame(preds)
print("\nval pred correlations:")
print(P.corr().round(3))
print("\nval pred describe:")
print(P.describe().loc[["mean","50%","std"]].round(2))

# OOF correlations of the 4 learners + tw
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw"]
print("\nOOF corr:")
print(oof[L].corr().round(3))
