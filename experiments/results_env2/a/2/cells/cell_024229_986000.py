
import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
# reconstruct E016 blend on OOF: find weights matching MAE 61.651
y = oof["y"].values
def mae(p): return np.abs(p-y).mean()
cands = {
 "0.4med_v3+0.3med_all+0.3hgbq_all": .4*oof.med_v3+.3*oof.med_all+.3*oof.hgbq_all,
 "0.4med_v3+0.3hgbq_v3+0.3hgbq_all": .4*oof.med_v3+.3*oof.hgbq_v3+.3*oof.hgbq_all,
 "0.4med_v3+0.6hgbq_all": .4*oof.med_v3+.6*oof.hgbq_all,
 "0.5med_v3+0.5hgbq_all": .5*oof.med_v3+.5*oof.hgbq_all,
}
for k,v in cands.items(): print(k, round(mae(v),3))

# simulate household-mean shrinkage with train-only hhmean, evaluated on later snapshots
oof["e016"] = cands["0.4med_v3+0.3med_all+0.3hgbq_all"]
for cut in [319, 347]:
    tr = oof[oof.snapshot_day<=cut]
    hh = tr.groupby("household_key")["e016"].mean().rename("hhmean")
    ev = oof[oof.snapshot_day>cut].merge(hh, on="household_key", how="left")
    ev["hhmean"] = ev["hhmean"].fillna(ev["e016"])
    print(f"--- cut {cut}: base MAE {np.abs(ev.e016-ev.y).mean():.3f}, n_hh_missing {ev.hhmean.isna().sum()}")
    for w in [0.2,0.3,0.4,0.5,0.6,0.7]:
        p = w*ev.hhmean + (1-w)*ev.e016
        print(f"  w={w}: MAE {np.abs(p-ev.y).mean():.3f}")
