
import agent_api as A
print(A.snapshot_days())
for name in ["season_demo_v1","mkt_v1","rfm_cadence_v1","rfm_v1","demo_v1","ewma_block_v1","comp_v1","rfm_traj_v1"]:
    try:
        df = A.load_saved(name + ".parquet")
        cols = list(df.columns)
        print(name, df.shape, "nfeat=", len(cols)-2)
        print(cols[:60])
        print(cols[60:130])
        print(cols[130:])
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
v = A.snapshot()
t = v.transactions
print(t[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity','trans_time']].describe())
r = v.table('coupon_redemptions')
print("redemptions", r.shape)
print(r.head())


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np

season = A.load_saved("season_demo_v1.parquet")   # E008: RFM+cadence+demo+seasonal (120)
mkt    = A.load_saved("mkt_v1.parquet")           # marketing block (21)
comp   = A.load_saved("comp_v1.parquet")          # competition block (21)
ewma   = A.load_saved("ewma_block_v1.parquet")    # ewma block (32)

def feats(df, exclude):
    return [c for c in df.columns if c not in exclude]

base_keys = ["household_key","snapshot_day"]
def add_block(df, block, tag, exclude):
    b = block.copy()
    dup = [c for c in b.columns if c in exclude]
    b = b[base_keys + [c for c in b.columns if c not in exclude]]
    b = b.rename(columns={c: c if c in base_keys else tag+c for c in b.columns})
    return df.merge(b, on=base_keys, how="left")

df = season.copy()
exclude = set(season.columns)
df = add_block(df, mkt.drop(columns=feats(mkt, set(mkt.columns)-set(season.columns))), "", exclude) if False else df
# safer: take only new columns from each block
for block, tag in [(mkt,"m_"), (comp,"c_"), (ewma,"e_")]:
    newcols = [c for c in block.columns if c not in season.columns and c not in base_keys]
    add = block[base_keys + newcols].copy()
    add = add.rename(columns={c: tag+c for c in newcols})
    df = df.merge(add, on=base_keys, how="left")

print("shape:", df.shape, "nfeat:", len(df.columns)-2)
print([c for c in df.columns if c.startswith(("m_","c_","e_"))])
print("null frac:", df.isna().mean().mean().round(4))
A.save_table(df, "combined_v1.parquet")


# ---- cell ----

import agent_api as A, pandas as pd
ewma = A.load_saved("ewma_block_v1.parquet")
mkt  = A.load_saved("mkt_v1.parquet")
comp = A.load_saved("comp_v1.parquet")
season = A.load_saved("season_demo_v1.parquet")
for nm, b in [("ewma",ewma),("mkt",mkt),("comp",comp)]:
    print(nm, b.shape)
    print(list(b.columns))
    print("---")


# ---- cell ----

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


# ---- cell ----

import agent_api as A, pandas as pd

season = A.load_saved("season_demo_v1.parquet")
mkt    = A.load_saved("mkt_v1.parquet")
comp   = A.load_saved("comp_v1.parquet")
ewma   = A.load_saved("ewma_block_v1.parquet")
KEYS = ["household_key","snapshot_day"]

df = season.copy()
def add(block, prefix, newcols):
    add_df = block[KEYS + newcols].rename(columns={c: prefix+c for c in newcols})
    return df.merge(add_df, on=KEYS, how="left")

mkt_new = [c for c in mkt.columns if c.startswith(("n_tgt","tgt_","days_since_tgt","n_red","days_since_red","disp_","mail_","sh_coupon","sh_retaildisc","sh_match"))]
df = add(mkt, "m_", mkt_new)
comp_new = [c for c in comp.columns if c.startswith(("sh_","fresh_share","private_share","national_share","coupon_disc_share","coupon_match_disc_share","retail_disc_share","n_stores","top_store_share","lines_per_basket","units_per_basket","evening_basket_share","trans_time_mean","top_dow_share","dep_entropy","n_depts"))]
df = add(comp, "c_", comp_new)
ewma_new = ["ew_spend_h14","ew_trips_h14","ew_spend_h28","ew_trips_h28","ew_spend_h56","ew_trips_h56","ew_spend_h112","ew_trips_h112","spend_7d","trips_7d","spend_14d","trips_14d","blk_mean","blk_median","blk_std","blk_min","blk_max","blk_cv","blk_active_frac","blk_active_frac6","blk_last_over_mean","blk_year_ago","blk_slope","blk_zero_run","n_stores_84d","active_days_28d","spend_per_active_day_28d","week_of_year","week_sin","week_cos"]
df = add(ewma, "e_", ewma_new)

print("shape:", df.shape, "nfeat:", len(df.columns)-2)
print("dup cols:", len(df.columns) - len(set(df.columns)))
print("null frac:", round(df.drop(columns=KEYS).isna().mean().mean(),4))
p = A.save_table(df, "union_all_v1.parquet")
print(p)
