import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
tt = A.train_targets()
print("oof rows", len(oof), "tt rows", len(tt))
m = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
print("y mismatch:", (m.y_x.fillna(-1) != m.y_y.fillna(-1)).sum() if "y_y" in m.columns else "cols:", m.columns.tolist())

learners = ["med_v3","med_all","hgbq_v3","hgbq_all"]
y = oof["y"].values
def mae(p): return np.mean(np.abs(p - y))
for c in learners:
    print(f"{c:9s} OOF MAE {mae(oof[c].values):.3f}")
# find E016 blend weights approx: try grid
best=(1e9,None)
import itertools
for w in itertools.product(np.arange(0,1.01,0.05), repeat=3):
    if w[0]+w[1]+w[2] > 1.001: continue
    w4 = 1 - w[0]-w[1]-w[2]
    if w4 < -0.001: continue
    p = w[0]*oof.med_v3 + w[1]*oof.med_all + w[2]*oof.hgbq_v3 + w4*oof.hgbq_all
    mm = mae(p.values)
    if mm < best[0]: best=(mm,w+(round(w4,2),))
print("best fixed-weight blend on OOF:", best)

# per-snapshot MAE of best blend
w = best[1]
p = w[0]*oof.med_v3 + w[1]*oof.med_all + w[2]*oof.hgbq_v3 + w[3]*oof.hgbq_all
oof["blend"] = p
print(oof.groupby("snapshot_day").apply(lambda g: pd.Series({"n":len(g),"y_med":g.y.median(),"mae_blend":np.mean(np.abs(g.blend-g.y)),"mae_medv3":np.mean(np.abs(g.med_v3-g.y))}), include_groups=False).round(2))

# NaN check other oofs
for nm in ["oof_tw","oof_pt","oof_harness"]:
    d = A.load_saved(nm+".parquet")
    col = [c for c in d.columns if c.startswith("oof")][0]
    print(nm, "NaN by day:", d.groupby("snapshot_day")[col].apply(lambda s: s.isna().mean().round(2)).to_dict())
