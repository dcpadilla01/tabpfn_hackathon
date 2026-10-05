import pandas as pd
e3 = agent_api.load_saved("e003_union.parquet")
mk = agent_api.load_saved("e005_marketing_only.parquet")
print("e3", e3.shape, "mk", mk.shape)
m = e3.merge(mk.drop(columns=[c for c in mk.columns if c in ("snapshot_day",)] and ["snapshot_day"] if False else []), on=["household_key","snapshot_day"], how="inner")
print("merged", m.shape, "dup cols check:", len(m.columns))
agent_api.save_table(m, "e005_marketing.parquet")
