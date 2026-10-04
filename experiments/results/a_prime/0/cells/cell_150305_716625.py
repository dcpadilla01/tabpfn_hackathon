import pandas as pd
for name in ["mkt_v2","hist_v2","season_v1","mix_v1","e006_temporal","e007_robust","structure_v1","e005_full_plus_mix"]:
    try:
        df = agent_api.load_saved(name+".parquet")
        print(name, df.shape)
        print(list(df.columns))
        print("---")
    except Exception as e:
        print(name, "ERR", e)

tt = agent_api.train_targets()
print(tt.shape, tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())
print(agent_api.snapshot_days())
