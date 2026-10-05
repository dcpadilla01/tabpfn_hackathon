import agent_api as api, pandas as pd, numpy as np
names = ["allF","oof_e5","repro_e5","e003_preds","e004_preds","e005_preds","e007_preds","e009_preds","e010_preds","e011_preds","e012_preds","e014_preds","e016_preds","e017_preds","e005_newfeats","e004_new","lagfeats","lagfeats2","f_weekly","camp_feats","e016_allpreds","e016_held","e016_sqpreds","e002_features"]
for nm in names:
    try:
        df = api.load_saved(nm + ".parquet")
        days = sorted(df['snapshot_day'].unique()) if 'snapshot_day' in df.columns else None
        print(f"{nm}: shape={df.shape} days={days} cols={list(df.columns)[:12]}")
    except Exception as e:
        print(f"{nm}: ERR {type(e).__name__} {e}")
