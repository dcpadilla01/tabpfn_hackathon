
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet").copy()
oof["resid"] = oof.y - oof.oof
f4 = agent_api.load_saved("feats_v4.parquet")
oof = oof.merge(f4[["household_key","snapshot_day","spend_28"]], on=["household_key","snapshot_day"])
oof["bin"] = pd.qcut(oof.oof, 10, duplicates="drop").astype(str)
snaps = sorted(oof.snapshot_day.unique())

def eval_stack(s, use_bin=True, use_clip=True, w=0.97, kappa=0.0, hh_shrink=False):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    pv = va.oof.values.copy(); yv = va.y.values
    if use_bin:
        corr = tr.groupby("bin").resid.median()
        pv = pv + va.bin.map(corr).astype(float).fillna(0).values
    if hh_shrink:
        g = tr.groupby("household_key").agg(my=("y","mean"), mp=("oof","mean"))
        g["r"] = (g.my/g.mp).clip(0.5,2.0)
        r = va.household_key.map(g.r).fillna(1.0).values**kappa
        pv = pv*r
    if use_clip: pv = np.clip(pv, 0, None)
    r28 = va.spend_28.fillna(0).values
    pv = w*pv + (1-w)*r28
    return np.abs(pv-yv).mean(), len(va)

# grid over stack options (honest LOO per snapshot)
res = []
for use_bin in [True, False]:
    for w in [0.95, 0.97, 1.0]:
        for kappa in [0.0, 0.5]:
            tot=0; n=0
            for s in snaps:
                m,k = eval_stack(s, use_bin, True, w, kappa, kappa>0)
                tot+=m*k; n+=k
            res.append((round(tot/n,3), use_bin, w, kappa))
for r in sorted(res): print(r)
