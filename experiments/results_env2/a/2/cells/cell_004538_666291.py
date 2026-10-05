import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")

feats = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
print("feats", feats.shape)
print("cols:", list(feats.columns))
y = tt.future_spend_4w
print("target mean %.1f med %.1f p90 %.1f p99 %.1f zero-rate %.3f" % (y.mean(), y.median(), y.quantile(.9), y.quantile(.99), (y==0).mean()))

oof = A.load_saved("oof_e008.parquet")
print("oof cols:", oof.columns.tolist())
m = oof.merge(tt, on=["household_key","snapshot_day"])
pc = [c for c in oof.columns if c not in ("household_key","snapshot_day")]
for c in pc:
    e = m[c]-m.future_spend_4w
    print(c, "OOF MAE %.2f bias %.2f" % (e.abs().mean(), e.mean()))
c0 = pc[0]
m["dec"] = pd.qcut(m[c0], 10, duplicates="drop")
for k, d in m.groupby("dec", observed=True):
    e = d[c0]-d.future_spend_4w
    print("dec %-12s n %5d pred %6.1f y %6.1f bias %7.1f mae %6.1f" % (str(k), len(d), d[c0].mean(), d.future_spend_4w.mean(), e.mean(), e.abs().mean()))
for sd, d in m.groupby("snapshot_day"):
    e = d[c0]-d.future_spend_4w
    print("snap", sd, "mae %.1f bias %.1f ymean %.1f" % (e.abs().mean(), e.mean(), d.future_spend_4w.mean()))
