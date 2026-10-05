import agent_api as A, pandas as pd, numpy as np
print("snapshot_days:", A.snapshot_days())
print("KEYS:", A.KEYS, "TARGET:", A.TARGET)
names = ["feats_v3","feats_seasonal","feats_e014","oof_e008","pred_e008","pred_e009","pred_e010_blend","pred_seasonal","pred_e014","pred_log1p","pred_e006","pred_e004","pred_e005"]
for n in names:
    try:
        df = A.load_saved(n + ".parquet")
        cols = list(df.columns)
        print("==", n, df.shape, "| ncols:", len(cols))
        print("   cols:", cols[:10], "..." if len(cols)>10 else "")
        if "snapshot_day" in cols:
            vc = df.snapshot_day.value_counts().sort_index()
            print("   days:", dict(vc))
    except Exception as e:
        print("==", n, "ERR", type(e).__name__, str(e)[:120])
tt = A.train_targets()
print("train_targets:", tt.shape, "zero share:", round(float((tt.future_spend_4w==0).mean()),4))
print(tt.future_spend_4w.describe())
v = A.snapshot()
tx = v.table("transactions")
print("transactions@459:", tx.shape)
