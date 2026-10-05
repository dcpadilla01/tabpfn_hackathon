import agent_api as A, pandas as pd, numpy as np
names = ["feats_all_e016","feats_e014","feats_prof","feats_seasonal","feats_v3","oof_e008","oof_e016_cv","oof_harness","oof_pt","past_targets",
         "pred_e001","pred_e002","pred_e003","pred_e004","pred_e005","pred_e006","pred_e007","pred_e008","pred_e009","pred_e010","pred_e010_blend",
         "pred_e014","pred_e016","pred_e018","pred_log1p","pred_prof","pred_pt","pred_seasonal"]
for n in names:
    try:
        df = A.load_saved(n+".parquet")
        print(n, df.shape, df.columns.tolist()[:12])
    except Exception as e:
        print(n, "ERR", type(e).__name__, str(e)[:80])
print(A.snapshot_days())
tt = A.train_targets()
print("train_targets", tt.shape, tt.columns.tolist())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]).round(2))
