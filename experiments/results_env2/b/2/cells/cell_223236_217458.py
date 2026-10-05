
import agent_api as A, pandas as pd

season = A.load_saved("season_demo_v1.parquet")
mkt    = A.load_saved("mkt_v1.parquet")
comp   = A.load_saved("comp_v1.parquet")
ewma   = A.load_saved("ewma_block_v1.parquet")
KEYS = ["household_key","snapshot_day"]

df = season.copy()
def add(block, prefix, drop_cols):
    newcols = [c for c in block.columns if c not in KEYS and c not in drop_cols and c not in df.columns]
    add = block[KEYS + newcols].rename(columns={c: prefix+c for c in newcols})
    return df.merge(add, on=KEYS, how="left")

# marketing block new features (exclude its copy of RFM base)
mkt_new = [c for c in mkt.columns if c.startswith(("n_tgt","tgt_","days_since_tgt","n_red","days_since_red","disp_","mail_","sh_coupon","sh_retaildisc","sh_match"))]
df = add(mkt[mkt_new], "m_", set())
# composition block: only dept/brand/store/basket/entropy features (exclude RFM, dem strings, marketing dupes)
comp_new = [c for c in comp.columns if c.startswith(("sh_","fresh_share","private_share","national_share","coupon_disc_share","coupon_match_disc_share","retail_disc_share","n_stores","top_store_share","lines_per_basket","units_per_basket","evening_basket_share","trans_time_mean","top_dow_share","dep_entropy","n_depts"))]
df = add(comp[comp_new], "c_", set())
# ewma block: only the 32 genuinely new trailing features
ewma_new = ["ew_spend_h14","ew_trips_h14","ew_spend_h28","ew_trips_h28","ew_spend_h56","ew_trips_h56","ew_spend_h112","ew_trips_h112","spend_7d","trips_7d","spend_14d","trips_14d","blk_mean","blk_median","blk_std","blk_min","blk_max","blk_cv","blk_active_frac","blk_active_frac6","blk_last_over_mean","blk_year_ago","blk_slope","blk_zero_run","n_stores_84d","active_days_28d","spend_per_active_day_28d","week_of_year","week_sin","week_cos"]
df = add(ewma[ewma_new], "e_", set())

print("shape:", df.shape, "nfeat:", len(df.columns)-2)
print("dup cols:", len(df.columns) - len(set(df.columns)))
print("null frac:", round(df.drop(columns=KEYS).isna().mean().mean(),4))
p = A.save_table(df, "union_all_v1.parquet")
print(p)
