import agent_api as A
import pandas as pd, numpy as np

base = A.load_saved("e005_marketing.parquet")
tt = A.train_targets().copy()
demo = A.snapshot(459).table("demographics")

spend_map = base[["household_key","snapshot_day","spend_4w"]]
edges = np.array([-1.0, 0.009, 25, 50, 100, 200, 500, 1e12])
def binof(s): return pd.cut(s, bins=edges, labels=False).astype(float)

tt2 = tt.merge(spend_map, on=["household_key","snapshot_day"], how="left")
tt2["bin"] = binof(tt2.spend_4w)

days = sorted(base.snapshot_day.unique())
outs = []
for d in days:
    past = tt2[tt2.snapshot_day < d]
    sub = base[base.snapshot_day == d][["household_key","spend_4w"]].copy()
    if len(past)==0:
        out = sub[["household_key"]].copy()
        for c in newc: out[c] = np.nan
    else:
        prior = past.future_spend_4w.mean()
        hh = past.groupby("household_key").future_spend_4w.agg(te_hh_mean="mean", te_hh_n="count")
        hh["te_hh_shrunk"] = (hh.te_hh_n*hh.te_hh_mean + 3*prior)/(hh.te_hh_n+3)
        b = past.groupby("bin").future_spend_4w.mean().rename("te_bin")
        dm = past.merge(demo, on="household_key", how="left")
        te_size = dm.groupby("classification_4").future_spend_4w.mean().rename("te_size")
        te_home = dm.groupby("homeowner_desc").future_spend_4w.mean().rename("te_home")
        te_kid  = dm.groupby("kid_category_desc").future_spend_4w.mean().rename("te_kid")
        out = sub.copy()
        out["bin"] = binof(out.spend_4w)
        out["te_prior"] = prior
        out = out.merge(hh.reset_index(), on="household_key", how="left")
        out = out.merge(b.reset_index(), on="bin", how="left")
        out = out.merge(demo[["household_key","classification_4","homeowner_desc","kid_category_desc"]], on="household_key", how="left")
        out = out.merge(te_size.reset_index(), on="classification_4", how="left")
        out = out.merge(te_home.reset_index(), on="homeowner_desc", how="left")
        out = out.merge(te_kid.reset_index(), on="kid_category_desc", how="left")
        out = out.drop(columns=["bin","classification_4","homeowner_desc","kid_category_desc","spend_4w"])
    outs.append(out)

newc = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid"]
te = pd.concat(outs, ignore_index=True)
print(te.shape)
print(te[newc].isna().mean().round(3).to_dict())
merged = base.merge(te[["household_key"]+newc], on="household_key", how="left")
print("merged", merged.shape, merged[newc].isna().mean().round(3).to_dict())
p = A.save_table(merged, "e007_te.parquet")
print("saved:", p)
