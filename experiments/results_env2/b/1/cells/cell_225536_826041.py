
import pandas as pd, numpy as np, agent_api

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)

feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    d = pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64)
print("design", Xn.shape)

sp28 = df["sp28"].values.astype(float)
bins = [0, 0.01, 25, 50, 100, 200, 400, 1e9]
labs = pd.cut(sp28, bins, labels=False)
print("\nE[y|sp28 bucket]:")
for b in sorted(pd.unique(labs)):
    m = labs==b
    print(f"  bucket {b}: n={m.sum():6d}  mean sp28={sp28[m].mean():7.1f}  mean y={y[m].mean():7.1f}  med y={np.median(y[m]):7.1f}")

snap = agent_api.snapshot(459)
tx = snap.transactions[["household_key","day","sales_value","week_no"]].copy()
print("\ntx rows:", len(tx), "neg sales:", (tx.sales_value<0).sum())

tx = tx.sort_values(["household_key","day"])
keys = tx.household_key.values
days = tx.day.values.astype(int)
sv = tx.sales_value.values.astype(float)
order = np.argsort(keys, kind="stable")
keys, days, sv = keys[order], days[order], sv[order]
uniq, start = np.unique(keys, return_index=True)
h2idx = {h:i for i,h in enumerate(uniq)}
ends = np.r_[start[1:], len(keys)]
cs = np.concatenate([[0.0], np.cumsum(sv)])

def win_sum(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    a = np.searchsorted(days[start[i]:ends[i]], lo, "left") + start[i]
    b = np.searchsorted(days[start[i]:ends[i]], hi, "right") + start[i]
    return cs[b]-cs[a]

hh = df["household_key"].values; sd = df["snapshot_day"].values
seas1y = np.array([win_sum(h, s-364, s-336) for h, s in zip(hh, sd)])
seas1y_b = np.array([win_sum(h, s-392, s-364) for h, s in zip(hh, sd)])
print("corr seas1y aligned:", round(np.corrcoef(seas1y, y)[0,1],3), "| misaligned:", round(np.corrcoef(seas1y_b, y)[0,1],3))

wk = tx.groupby("week_no").sales_value.sum()
wk_cnt = tx.groupby("week_no").household_key.nunique()
wk_per_hh = (wk / wk_cnt).fillna(0)
seas_idx, mkt_now, mkt_yoy = [], [], []
for s in sd:
    fw = list(range((s+9)//7, (s+28+8)//7 + 1))
    prev  = [wk_per_hh[w-52] for w in fw if (w-52) in wk_per_hh.index]
    prev2 = [wk_per_hh[w-104] for w in fw if (w-104) in wk_per_hh.index]
    seas_idx.append(np.mean(prev) if prev else np.nan)
    mkt_yoy.append(np.mean(prev)/np.mean(prev2) if prev and prev2 and np.mean(prev2)>0 else np.nan)
    now = [wk_per_hh[w] for w in range((s+8)//7-8, (s+8)//7) if w in wk_per_hh.index]
    mkt_now.append(np.mean(now) if now else np.nan)
seas_idx = np.array(seas_idx); mkt_now = np.array(mkt_now); mkt_yoy = np.array(mkt_yoy)
for s in np.unique(sd):
    m = sd==s
    print(f"  s={s}: seas_idx={np.nanmean(seas_idx[m]):6.1f} mkt_now={np.nanmean(mkt_now[m]):6.1f} yoy={np.nanmean(mkt_yoy[m]):5.2f} mean_y={y[m].mean():6.1f}")
