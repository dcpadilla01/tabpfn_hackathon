import agent_api, pandas as pd, numpy as np
pd.set_option("display.width", 200)
print(agent_api.snapshot_days())
for name in ["feats_v3","feats_seasonal"]:
    df = agent_api.load_saved(name+".parquet")
    print("==", name, df.shape)
    print(list(df.columns))
for name in ["pred_seasonal","pred_e008","pred_e010_blend","oof_e008","pred_e009"]:
    df = agent_api.load_saved(name+".parquet")
    print("==", name, df.shape, list(df.columns))
    print(df.head(3))
tt = agent_api.train_targets()
print("== train_targets", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())
import xgboost, sklearn
print("xgb", xgboost.__version__, "sklearn", sklearn.__version__)
