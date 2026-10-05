
import pandas as pd, numpy as np
import agent_api as A

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left").dropna(subset=["spend_28"])

y = m["future_spend_4w"].values

def mae(p): return np.mean(np.abs(y - p))

print("naive baselines on TRAIN rows:")
for c in ["spend_28","spend_84","spend_56","d_ewma_spend_hl28","d_ewma_spend_hl56","avg28_all","blk_1","decay_mean"]:
    print(f"  {c:22s} MAE={mae(m[c].values):8.3f}")
print("  pred=0                 MAE=%8.3f" % mae(np.zeros(len(y))))
print("  pred=median            MAE=%8.3f" % mae(np.full(len(y), np.median(y))))

# ratio y / spend_84 by bucket of spend_84
b = pd.qcut(m["spend_84"], 10, duplicates="drop")
g = m.groupby(b, observed=True).apply(lambda d: pd.Series({
    "n": len(d), "y_med": d.future_spend_4w.median(), "s84_med": d.spend_84.median(),
    "ratio_med": (d.future_spend_4w/ d.spend_84.clip(lower=1e-9)).median()}), include_groups=False)
print("\nmedian y and y/spend_84 by spend_84 decile:")
print(g.round(2).to_string())

# zero-target households: what do they look like?
z = m[m.future_spend_4w==0]; nz = m[m.future_spend_4w>0]
print(f"\nzero-target: {len(z)} ({len(z)/len(m):.1%})")
for c in ["spend_28","spend_84","recency","trips_28","d_ewma_spend_hl28"]:
    print(f"  {c:18s} zero: med={z[c].median():8.2f}   nonzero: med={nz[c].median():8.2f}")

# among zero-target, share with spend_28==0
print("zero-target & spend_28==0 share:", (z.spend_28==0).mean().round(3))
print("nonzero-target & spend_28==0 share:", (nz.spend_28==0).mean().round(3))

# log-space correlation
ly = np.log1p(y)
for c in ["spend_28","spend_84","d_ewma_spend_hl28","d_ewma_spend_hl56"]:
    print(f"corr log1p({c}) vs log1p(y): {np.corrcoef(np.log1p(m[c].values), ly)[0,1]:.3f}")
