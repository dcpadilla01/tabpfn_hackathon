
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet").copy()
oof["resid"] = oof.y - oof.oof
y = oof.y.values; p = oof.oof.values
print("base OOF MAE %.3f | pred min %.1f max %.1f" % (np.abs(p-y).mean(), p.min(), p.max()))

# A) clip negatives
pc = np.clip(p, 0, None)
print("A clip@0 MAE %.3f" % np.abs(pc-y).mean())

# B) per-pred-decile median-resid correction, fit on OTHER snapshots (LOO)
oof["bin"] = pd.qcut(p, 10, duplicates="drop").astype(str)
tot=np.zeros(3); n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    corr = tr.groupby("bin").resid.median()
    adj = va.bin.map(corr).astype(float).fillna(0).values
    pv = va.oof.values
    tot[0]+=np.abs(pv+adj-va.y.values).sum()
    tot[1]+=np.abs(np.clip(pv+adj,0,None)-va.y.values).sum()
    tot[2]+=np.abs(pv-va.y.values).sum(); n+=len(va)
print("B bin-corr MAE %.3f | +clip %.3f | base %.3f" % (tot[0]/n, tot[1]/n, tot[2]/n))

# C) aggregate bias: OOF mean vs actual mean per snapshot
print("\nC per-snapshot mean: actual vs oof")
g = oof.groupby("snapshot_day").agg(y=("y","mean"), p=("oof","mean"))
g["ratio"] = (g.y/g.p).round(3)
print(g.round(1))

# D) oracle household fixed effect (cheating diagnostic): predict hh mean of y over other snapshots
tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s].groupby("household_key").y.mean()
    va = oof[oof.snapshot_day==s]
    pr = va.household_key.map(tr).fillna(va.y.mean()).values
    tot += np.abs(pr-va.y.values).sum(); n+=len(va)
print("\nD oracle hh-FE MAE %.3f (vs model 61.5)" % (tot/n))

# E) LOO blend with spend_28
f4 = agent_api.load_saved("feats_v4.parquet")
oof = oof.merge(f4[["household_key","snapshot_day","spend_28"]], on=["household_key","snapshot_day"])
tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    r = va.spend_28.fillna(0).values
    best=(1e9,1)
    for w in np.arange(0.85,1.001,0.01):
        m = np.abs(w*va.oof.values+(1-w)*r-va.y.values).mean()
        if m<best[0]: best=(m,w)
    tot+=best[0]*len(va); n+=len(va)
print("E LOO blend spend_28 MAE %.3f" % (tot/n))
