
import agent_api, pandas as pd, numpy as np
names = ["feats_all_e016","feats_e014","feats_prof","feats_seasonal","feats_v1","feats_v2","feats_v3","feats_v4","feats_v5","feats_v6",
         "oof_e008","oof_e016_cv","oof_harness","oof_pt","past_targets",
         "pred_e008","pred_e009","pred_e010","pred_e010_blend","pred_e014","pred_e016","pred_log1p","pred_prof","pred_pt","pred_seasonal"]
for n in names:
    try:
        df = agent_api.load_saved(n + ".parquet")
        print("==", n, df.shape, list(df.columns))
        print(df.head(2).to_string())
    except Exception as e:
        print("==", n, "ERR", type(e).__name__, e)
print("snapshot_days:", agent_api.snapshot_days())
tt = agent_api.train_targets()
print("train_targets:", tt.shape, tt.columns.tolist())
print(tt.head(3).to_string())
