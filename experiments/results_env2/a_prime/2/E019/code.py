
import agent_api as A
t18 = A.load_saved("e018_tree_feats.parquet")
t17 = A.load_saved("e017_xsec_rank.parquet")
print("E018 cols:", list(t18.columns))
print("E018 shape:", t18.shape)
print("E017 shape:", t17.shape)
print("E017 cols:", list(t17.columns))
print("E018 snapshot days:", sorted(t18.snapshot_day.unique()))
print("E017 snapshot days:", sorted(t17.snapshot_day.unique()))


# ---- cell ----

import agent_api as A
import pandas as pd

t17 = A.load_saved("e017_xsec_rank.parquet")
t18 = A.load_saved("e018_tree_feats.parquet")

new_cols = ["max_basket_28","zero_wk_share_12","camp_TypeA","camp_TypeB","camp_TypeC",
            "camp_n","s28_x_dsl","s28_x_campA","ew28_x_dsl","disp_exposure"]
add = t18[["household_key","snapshot_day"] + new_cols]
print(add.dtypes)
print(add.isna().mean())

m = t17.merge(add, on=["household_key","snapshot_day"], how="left")
print("merged shape:", m.shape)
print("days:", sorted(m.snapshot_day.unique()))
print("n feature cols:", m.shape[1]-2)
print("dup keys:", m.duplicated(["household_key","snapshot_day"]).sum())
path = A.save_table(m, "e019_e017_plus_marketing.parquet")
print(path)
