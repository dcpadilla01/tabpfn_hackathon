import pandas as pd, numpy as np, agent_api
for name in ["e004_marketing_demo","e005_trend_season","e006_zero_inflation","e007_ar_lags","e008_log_transform","e009_full","e010_store_mix"]:
    t = agent_api.load_saved(name+".parquet")
    print(name, t.shape)
print()
print("e006_zero cols:", sorted(agent_api.load_saved("e006_zero_inflation.parquet").columns.tolist()))
print()
print("e005 cols:", sorted(agent_api.load_saved("e005_trend_season.parquet").columns.tolist()))
