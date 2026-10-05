
import agent_api, pandas as pd, numpy as np
for n in ["oof_e008","oof_e016_cv","oof_harness","oof_pt"]:
    df = agent_api.load_saved(n + ".parquet")
    print("==", n, df.shape, df.columns.tolist())
    print(df.groupby("snapshot_day").agg({"oof_sq":"mean","oof_med":"mean","oof_log":"mean"}).to_string() if "oof_sq" in df.columns else
          df.groupby("snapshot_day")["oof"].agg(["count","mean"]).to_string())
tt = agent_api.train_targets()
m = tt.merge(agent_api.load_saved("oof_e016_cv.parquet"), on=["household_key","snapshot_day"], how="left")
print("oof_e016_cv merged:", m["oof"].notna().mean())
print(m.dropna(subset=["oof"]).groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g.oof).mean()))
