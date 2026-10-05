import agent_api as A, pandas as pd, numpy as np
oo = A.load_saved("oof_e008.parquet")
tt = A.train_targets()
m = tt.merge(oo, on=["household_key","snapshot_day"])
m["blend"] = 0.25*m.oof_sq + 0.25*m.oof_med + 0.5*m.oof_log
y = m.future_spend_4w.values
for c in ["oof_sq","oof_med","oof_log","blend"]:
    e = np.abs(m[c].values - y)
    print(c, "OOF MAE", round(np.mean(e),3))
print("zero targets share:", (y==0).mean())
for mask,name in [((y==0).values,"zero"), ((y>0).values,"nonzero"), ((y>200).values,">200"), ((y>500).values,">500")]:
    e = np.abs(m["blend"].values[mask] - y[mask])
    print(name, "n", mask.sum(), "MAE", round(np.mean(e),2), "share of total abs err", round(np.sum(e)/np.sum(np.abs(m.blend.values-y)),3))
# error by snapshot
for d,g in m.groupby("snapshot_day"):
    print(d, len(g), round(np.mean(np.abs(g.blend-g.future_spend_4w)),2), "mean y", round(g.future_spend_4w.mean(),1))
