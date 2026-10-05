import agent_api as A
import pandas as pd, numpy as np

names = ["feats_all_e016","feats_v3","feats_seasonal","oof_e016_cv","oof_e008","oof_harness","oof_pt","oof_tw","past_targets",
"pred_e006","pred_e008","pred_e009","pred_e010","pred_e010_blend","pred_e014","pred_e016","pred_e018","pred_log1p","pred_prof","pred_pt","pred_seasonal"]
for nm in names:
    try:
        df = A.load_saved(nm + ".parquet")
        nonnum = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        print(f"{nm:20s} {str(df.shape):14s} cols={list(df.columns)[:12]} nonnum={nonnum[:4]}")
    except Exception as e:
        print(nm, "ERR", type(e).__name__, str(e)[:80])

print()
print("snapshot_days:", A.snapshot_days())
tt = A.train_targets()
print("train_targets", tt.shape, tt.columns.tolist())
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(2))

for nm in ["oof_e016_cv","oof_e008","oof_harness","oof_pt","oof_tw"]:
    df = A.load_saved(nm + ".parquet")
    print("\n--", nm, df.columns.tolist())
    print(df.head(2))
    if "snapshot_day" in df.columns:
        print("days:", sorted(df.snapshot_day.unique()))
