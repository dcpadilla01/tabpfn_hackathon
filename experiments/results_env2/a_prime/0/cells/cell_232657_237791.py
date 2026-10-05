import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")  # closure variable

def fn(view, snapshot_day):
    info = {}
    tx = view.table("transactions")
    info["view_day"] = int(view.day)
    info["tx_max_day"] = int(tx.day.max()) if len(tx) else -1
    info["n_hh"] = len(view.households)
    b = saved[saved.snapshot_day == snapshot_day].set_index("household_key")
    info["closure_rows"] = len(b)
    out = b[["spend_4w_recent"]].copy()
    out["view_day_probe"] = float(view.day)
    print(snapshot_day, info)
    return out

t = api.build_features(fn)
print(t.shape, t.columns.tolist())
