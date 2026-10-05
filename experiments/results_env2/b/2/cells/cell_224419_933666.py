import pandas as pd
base = load_saved("churn_vol_v1.parquet")
blk = load_saved("selfcal_v1.parquet")
out = base.merge(blk, on=["household_key","snapshot_day"], how="left")
print(out.shape)
save_table(out, "e011_table.parquet")
