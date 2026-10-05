
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
