import agent_api as api, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
saved = api.load_saved("e013_storeprod.parquet")
snap = api.snapshot(); tx_all = snap.transactions
for s in sorted(saved.snapshot_day.unique()):
    tx = tx_all[tx_all.day <= s]
    w84 = tx[tx.day >= s-83]
    hs = w84.groupby(["household_key","store_id"]).sales_value.sum().unstack(fill_value=0.0).reindex(index=saved[saved.snapshot_day==s].household_key.values, fill_value=0.0)
    print(s, "tx", len(tx), "w84", len(w84), "hs.shape", hs.shape, "argmax ok" if hs.shape[1]>0 else "EMPTY COLS")
    if hs.shape[1]==0:
        print("  w84 days:", w84.day.min() if len(w84) else None, w84.day.max() if len(w84) else None)
        print("  tx day range:", tx.day.min(), tx.day.max())
