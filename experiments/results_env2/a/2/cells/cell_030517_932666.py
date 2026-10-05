import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
y = oof["y"].values
L = ["med_v3","med_all","hgbq_v3","hgbq_all"]
P = oof[L].values
def mae_w(w): return np.abs(y - P @ np.array(w)).mean()
# candidate E016 weight vectors
cands = {"0.4/0.3/0.3/0(med_v3,med_all,hgbq_v3)": [0.4,0.3,0.3,0.0],
         "0.4/0/0.3/0.3": [0.4,0.0,0.3,0.3],
         "0.4/0.3/0/0.3": [0.4,0.3,0.0,0.3],
         "0.4/0/0.3/0.3 alt": [0.4,0.0,0.3,0.3]}
for k,w in cands.items(): print(k, round(mae_w(w),4))

# check oof_e008 provenance
oe = A.load_saved("oof_e008.parquet")
print("\noof_e008 snapshot days:", sorted(oe.snapshot_day.unique()), "n rows", len(oe))

# feature tables: is feats_all_e016 == v3 + seasonal?
v3 = A.load_saved("feats_v3.parquet"); seas = A.load_saved("feats_seasonal.parquet"); allf = A.load_saved("feats_all_e016.parquet")
key = ["household_key","snapshot_day"]
m = v3.merge(seas, on=key, how="outer", suffixes=("_v3","_s"))
print("\nv3 cols:", len(v3.columns)-2, "seasonal cols:", len(seas.columns)-2, "merged:", m.shape)
common = [c for c in allf.columns if c in m.columns]
print("all_e016 cols:", len(allf.columns)-2)
print("cols in all_e016 not in v3+seasonal:", [c for c in allf.columns if c not in v3.columns and c not in seas.columns and c not in key])
# verify equality on a few cols
chk = allf.merge(m, on=key, suffixes=("_a","_m"))
for c in ["spend_all","lag364_spend"]:
    if c+"_a" in chk.columns and c+"_m" in chk.columns:
        print(c, "max abs diff:", np.abs(chk[c+"_a"]-chk[c+"_m"]).max())
print("seasonal cols list:", [c for c in seas.columns if c not in key])
