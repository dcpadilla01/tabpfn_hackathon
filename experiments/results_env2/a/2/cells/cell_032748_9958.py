import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
oof = oof.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
fv = A.load_saved("feats_v3.parquet")
oof = oof.merge(fv[["household_key","snapshot_day","spend_28","spend_56","spend_84","baskets_84","baskets_28","spend_all"]],
                on=["household_key","snapshot_day"], how="left")
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[L].values @ W0
tr = oof.y.notna()
d = oof[tr].copy()
y = d.y.values; b = d.blend.values
base = np.mean(np.abs(b-y)); print("train MAE blend:", round(base,3))

for col in ["spend_84","spend_56","spend_28"]:
    z = d[col].values == 0
    print(f"\nrule {col}==0: n={z.sum()} ({z.mean():.1%})")
    print("  of which y==0:", ((y==0)&z).sum(), " y>0:", ((y>0)&z).sum())
    if z.sum():
        p = np.where(z, 0.0, b)
        print("  MAE after zeroing:", round(np.mean(np.abs(p-y)),3), " delta:", round(np.mean(np.abs(p-y))-base,3))
        # partial: only zero if blend below threshold t
        for t in [10,25,50]:
            zz = z & (b<t)
            p = np.where(zz, 0.0, b)
            print(f"    zero if {col}==0 & blend<{t}: n={zz.sum()}, MAE={np.mean(np.abs(p-y)):.3f}, delta={np.mean(np.abs(p-y))-base:+.3f}")

# also: what does blend predict on true-zero rows with spend_84==0?
m = (y==0) & (d.spend_84.values==0)
print("\ntrue-zero & spend_84==0: n=", m.sum(), " blend mean:", b[m].mean().round(1), " median:", np.median(b[m]).round(1))
m2 = (y>0) & (d.spend_84.values==0)
print("y>0 & spend_84==0: n=", m2.sum(), " y mean:", y[m2].mean().round(1), " blend mean:", b[m2].mean().round(1))
# how many val rows would be zeroed?
va = ~tr
zv = oof.loc[va,"spend_84"].values==0
print("\nval rows spend_84==0:", zv.sum(), "of", va.sum())
