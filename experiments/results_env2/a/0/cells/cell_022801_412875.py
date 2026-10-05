
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet")
y = oof.y.values; p = oof.oof.values
base = np.abs(p-y).mean()
print("base OOF MAE %.3f" % base)

# median resid by pred-bin (out-of-fold per snapshot)
oof["bin"] = pd.qcut(p, 10, duplicates="drop")
print("\nmedian resid per bin:")
print(oof.groupby("bin", observed=True).agg(p=("oof","median"), medres=("resid","median"), n=("y","size")).round(1))

# grid: global affine a*p+b (MAE-optimal shrink)
best=(1e9,None)
for a in np.arange(0.90,1.11,0.02):
    for b in range(-6,7,2):
        m = np.abs(a*p+b-y).mean()
        if m<best[0]: best=(m,(round(a,2),b))
print("\nbest affine:", best)

# blend with raw predictors
f4 = agent_api.load_saved("feats_v4.parquet")
oof2 = oof.merge(f4[["household_key","snapshot_day","exp4w_blend","lag1_spend","spend_28","wk_mean8"]], on=["household_key","snapshot_day"])
for col in ["exp4w_blend","lag1_spend","spend_28","wk_mean8"]:
    r = oof2[col].fillna(0).values
    print("%s alone MAE %.3f" % (col, np.abs(r-y).mean()))
    bestw=(1e9,None)
    for w in np.arange(0.6,1.01,0.05):
        m = np.abs(w*p+(1-w)*r-y).mean()
        if m<bestw[0]: bestw=(m,round(w,2))
    print("  best blend w=%.2f MAE %.3f" % (bestw[1], bestw[0]))

# per-bin additive median correction (fit on other snapshots, LOO)
tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    corr = tr.groupby("bin", observed=True).resid.median()
    adj = va.bin.map(corr).fillna(0).values
    tot += np.abs(va.oof.values+adj-va.y.values).sum(); n += len(va)
print("\nper-bin median correction MAE %.3f" % (tot/n))
