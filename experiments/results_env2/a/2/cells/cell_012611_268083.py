import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values
med = df.oof_med.values; sq = df.oof_sq.values
print("base MAE med", round(np.abs(med-y).mean(),3), " sq", round(np.abs(sq-y).mean(),3))
for off in [0,5,10,15,17,20,25,30]:
    print("offset",off, "med:", round(np.abs(med+off-y).mean(),3), " sq:", round(np.abs(sq+off-y).mean(),3))
# per-snapshot optimal offset for med
print("\nper-snapshot optimal offset (med):")
for d,g in df.groupby("snapshot_day"):
    yy=g.future_spend_4w.values; mm=g.oof_med.values
    offs=np.arange(0,60,1); maes=[np.abs(mm+o-yy).mean() for o in offs]
    i=int(np.argmin(maes))
    print(d, "n=",len(g), "opt_off=",offs[i], "mae=",round(maes[i],2), "base=",round(np.abs(mm-yy).mean(),2), "mean_bias=",round((mm-yy).mean(),2))
# offset by spend_28 segment (need feats)
f3 = agent_api.load_saved("feats_v3.parquet")
df2 = df.merge(f3[["household_key","snapshot_day","spend_28"]], on=["household_key","snapshot_day"])
print("\nper spend_28 bin optimal offset (med):")
bins=[-1,1,50,150,300,1e9]
df2["b"]=pd.cut(df2.spend_28,bins,labels=["0","low","mid","high","vhigh"])
for b,g in df2.groupby("b",observed=True):
    yy=g.future_spend_4w.values; mm=g.oof_med.values
    offs=np.arange(0,80,2); maes=[np.abs(mm+o-yy).mean() for o in offs]
    i=int(np.argmin(maes))
    print(b, "n=",len(g), "opt_off=",offs[i], "mae=",round(maes[i],2), "base=",round(np.abs(mm-yy).mean(),2))
