import agent_api as api, pandas as pd
snap = api.snapshot()
tx = snap.transactions
print("len:", len(tx), "maxday:", tx.day.max() if len(tx) else None)
tx2 = snap.table("transactions")
print("len2:", len(tx2))
h95 = api.load_saved("e013_storeprod.parquet")
h95 = h95[h95.snapshot_day==95].household_key.values[:5]
print(h95)
h = api.history(h95[0])
print("hist rows:", len(h), "days:", h.day.min() if len(h) else None, h.day.max() if len(h) else None)
