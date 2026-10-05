
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet")
oof["resid"] = oof.y - oof.oof
y = oof.y.values; p = oof.oof.values
print("base OOF MAE %.3f" % np.abs(p-y).mean())

oof["bin"] = pd.qcut(p, 10, duplicates="drop")
print("median resid per bin:")
print(oof.groupby("bin", observed=True).agg(p=("oof","median"), medres=("resid","median"), n=("y","size")).round(1))

best=(1e9,None)
for a in np.arange(0.90,1.11,0.02):
    for b in range(-6,7,2):
        m = np.abs(a*p+b-y).mean()
        if m<best[0]: best=(m,(round(a,2),b))
print("best affine:", best)

f4 = agent_api.load_saved("feats_v4.parquet")
oof2 = oof.merge(f4[["household_key","snapshot_day","exp4w_blend","lag1_spend","spend_28","wk_mean8"]], on=["household_key","snapshot_day"])
for col in ["exp4w_blend","lag1_spend","spend_28","wk_mean8"]:
    r = oof2[col].fillna(0).values
    bestw=(1e9,None)
    for w in np.arange(0.6,1.01,0.05):
        m = np.abs(w*p+(1-w)*r-y).mean()
        if m<bestw[0]: bestw=(m,round(w,2))
    print("%s: alone MAE %.3f | best blend w=%.2f MAE %.3f" % (col, np.abs(r-y).mean(), bestw[1], bestw[0]))

tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    corr = tr.groupby("bin", observed=True).resid.median()
    adj = va.bin.map(corr).fillna(0).values
    tot += np.abs(va.oof.values+adj-va.y.values).sum(); n += len(va)
print("per-bin median correction MAE %.3f" % (tot/n))
