import agent_api as A, pandas as pd, numpy as np
oo = A.load_saved("oof_e008.parquet")
tt = A.train_targets()
m = tt.merge(oo, on=["household_key","snapshot_day"])
m["blend"] = 0.25*m["oof_sq"] + 0.25*m["oof_med"] + 0.5*m["oof_log"]
y = m["future_spend_4w"].values
abs_err = np.abs(m["blend"].values - y)
tot = abs_err.sum()
for mask,name in [(y==0,"zero"), (y>200,">200"), (y>500,">500"), (y>1000,">1000")]:
    e = abs_err[mask]
    print(name, "n", mask.sum(), "MAE", round(np.mean(e),2), "err share", round(e.sum()/tot,3))
best=(None,1e9)
for w in np.arange(0,1.01,0.05):
    for v in np.arange(0,1.01-w,0.05):
        p = w*m["oof_sq"]+v*m["oof_med"]+(1-w-v)*m["oof_log"]
        e = np.mean(np.abs(p-y))
        if e<best[1]: best=((w,v,1-w-v),e)
print("best OOF blend:", best)
for d,g in m.groupby("snapshot_day"):
    print(d, len(g), round(np.mean(np.abs(g["blend"]-g["future_spend_4w"])),2), "mean y", round(g["future_spend_4w"].mean(),1))
