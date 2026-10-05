import agent_api, pandas as pd, numpy as np
pd.set_option("display.width", 250)

t = agent_api.load_saved("e013_stationary.parquet")
print("E013 shape:", t.shape)
print("E013 cols:", list(t.columns))

tt = agent_api.train_targets()
print("\nTarget by train snapshot:")
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(1))

snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby("week_no").agg(spend=("sales_value","sum"), baskets=("basket_id","nunique"), hhs=("household_key","nunique"))
g["spend_per_basket"] = g.spend/g.baskets
print("\nWeekly aggregates (week_no, spend, baskets, hhs, spend/basket):")
print(g.round(1).to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

# ---- 1) YoY seasonality check on weekly panel spend ----
snap = agent_api.snapshot(459)
tx = snap.transactions
wk = tx.groupby("week_no").sales_value.sum()
w = wk.reindex(range(1,67))
a = w.loc[1:50].values; b = w.loc[53:102].values[:50] if len(wk)>52 else None
wk2 = tx.groupby("week_no").sales_value.sum()
print("weeks available:", wk2.index.min(), wk2.index.max())
if wk2.index.max() >= 66:
    x = wk2.loc[1:14].values; y = wk2.loc[53:66].values
    print("corr weeks1-14 vs 53-66:", np.corrcoef(x,y)[0,1].round(3))
# detrended autocorr at lag 52 using log spend minus rolling mean
ls = np.log(wk2.values)
lag52 = ls[:-52]; cur = ls[52:]
print("n pairs lag52:", len(lag52), "corr:", np.corrcoef(lag52, cur)[0,1].round(3))
# lag 4 (monthly) and lag 13 (quarterly)
for L in [4,13,26,52]:
    if len(ls)>L:
        print(f"autocorr lag{L}:", np.corrcoef(ls[:-L], ls[L:])[0,1].round(3))

# ---- 2) feature/target diagnostics on train rows ----
t = agent_api.load_saved("e013_stationary.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"], how="left")
y = m.future_spend_4w
print("\ntrain rows:", len(m), "target zero share:", (y==0).mean().round(3), "target p50/p90/p99:", np.percentile(y,[50,90,99]).round(1))
feats = [c for c in t.columns if c not in ("household_key","snapshot_day")]
num = m[feats].select_dtypes(include=[np.number])
cors = {}
for c in num.columns:
    v = num[c]
    ok = v.notna()
    if ok.sum()>100:
        cors[c] = np.corrcoef(v[ok], y[ok])[0,1]
cs = pd.Series(cors).sort_values(key=np.abs, ascending=False)
print("\ntop |corr| with target:")
print(cs.head(25).round(3).to_string())
print("\nweakest |corr|:")
print(cs.tail(12).round(3).to_string())

# ---- 3) mean-reversion test ----
lp = lambda s: np.log1p(s)
base = lp(m.spend_84/3.0)
spike = lp(m.spend_28) - base
print("\ncorr(target, log spend_28):", np.corrcoef(lp(m.spend_28).fillna(0), y)[0,1].round(3))
print("corr(target, spike=log s28 - log s84/3):", np.corrcoef(spike.fillna(0), y)[0,1].round(3))
X = np.column_stack([lp(m.spend_28).fillna(0), spike.fillna(0)])
beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
print("bivariate coefs [log s28, spike]:", beta.round(2))
# same on log target scale for nonzero
nz = y>0
X2 = np.column_stack([lp(m.spend_28).fillna(0), spike.fillna(0)])[nz]
b2, *_ = np.linalg.lstsq(X2, np.log1p(y[nz]).values, rcond=None)
print("log-target coefs [log s28, spike]:", b2.round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved("e013_stationary.parquet")
print("base:", base.shape)

def fn(view, sd):
    tx = view.table("transactions")
    idx = view.households.index
    lg = np.log1p
    def wsum(w):
        return tx[tx.day > sd - w].groupby("household_key").sales_value.sum()
    s7, s14, s28, s56, s84 = (wsum(w) for w in (7, 14, 28, 56, 84))
    age = (sd - tx.day).astype(float)
    def dec(hl):
        return pd.Series(tx.sales_value.values * np.power(0.5, age.values / hl),
                         index=tx.index).groupby(tx.household_key).sum()
    d28, d112 = dec(28), dec(112)
    R = lambda s: s.reindex(idx).fillna(0.0)
    s7, s14, s28, s56, s84, d28, d112 = map(R, (s7, s14, s28, s56, s84, d28, d112))
    f = pd.DataFrame(index=idx)
    f["spike_7"]  = lg(s7)  - lg(s84 * 7 / 84)
    f["spike_14"] = lg(s14) - lg(s84 * 14 / 84)
    f["spike_28"] = lg(s28) - lg(s84)
    f["spike_56"] = lg(s56) - lg(s84 * 56 / 84)
    f["spike_dec"] = lg(d28) - lg(d112 * 28 / 112)
    f["spike_28_pos"] = f["spike_28"].clip(lower=0)
    f["spike_28_neg"] = f["spike_28"].clip(upper=0)
    rev = d28 * np.exp(-0.7 * f["spike_28_pos"])
    f["rev_rate_28"] = np.where(d28 > 0, rev, 0.0)
    return f

out = agent_api.build_features(fn)
print("new feats:", out.shape, list(out.columns))
merged = base.merge(out, on=["household_key", "snapshot_day"], how="left")
print("merged:", merged.shape, "NaN frac:", round(merged.isna().mean().mean(), 4))
print(merged[["spike_28","spike_dec","rev_rate_28"]].describe().round(3))
path = agent_api.save_table(merged, "e017_reversion.parquet")
print("saved:", path)


# ---- cell ----
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved("e013_stationary.parquet")

def fn(view, sd):
    tx = view.table("transactions")
    idx = view.households
    lg = np.log1p
    def wsum(w):
        return tx[tx.day > sd - w].groupby("household_key").sales_value.sum()
    s7, s14, s28, s56, s84 = (wsum(w) for w in (7, 14, 28, 56, 84))
    age = (sd - tx.day).astype(float)
    def dec(hl):
        return pd.Series(tx.sales_value.values * np.power(0.5, age.values / hl),
                         index=tx.index).groupby(tx.household_key).sum()
    d28, d112 = dec(28), dec(112)
    R = lambda s: s.reindex(idx).fillna(0.0)
    s7, s14, s28, s56, s84, d28, d112 = map(R, (s7, s14, s28, s56, s84, d28, d112))
    f = pd.DataFrame(index=idx)
    f["spike_7"]  = lg(s7)  - lg(s84 * 7 / 84)
    f["spike_14"] = lg(s14) - lg(s84 * 14 / 84)
    f["spike_28"] = lg(s28) - lg(s84)
    f["spike_56"] = lg(s56) - lg(s84 * 56 / 84)
    f["spike_dec"] = lg(d28) - lg(d112 * 28 / 112)
    f["spike_28_pos"] = f["spike_28"].clip(lower=0)
    f["spike_28_neg"] = f["spike_28"].clip(upper=0)
    rev = d28 * np.exp(-0.7 * f["spike_28_pos"])
    f["rev_rate_28"] = np.where(d28 > 0, rev, 0.0)
    return f

out = agent_api.build_features(fn)
print("new feats:", out.shape, list(out.columns))
merged = base.merge(out, on=["household_key", "snapshot_day"], how="left")
print("merged:", merged.shape, "NaN frac:", round(merged.isna().mean().mean(), 4))
print(merged[["spike_28","spike_dec","rev_rate_28"]].describe().round(3))
path = agent_api.save_table(merged, "e017_reversion.parquet")
print("saved:", path)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved("e013_stationary.parquet")
t = agent_api.load_saved("e017_reversion.parquet")
print(base.dtypes.head(3)); print(t.dtypes.head(3))
print("base hk dtype:", base.household_key.dtype, "t hk dtype:", t.household_key.dtype)
print("base sd dtype:", base.snapshot_day.dtype, "t sd dtype:", t.snapshot_day.dtype)
nan_by_col = t.isna().mean().sort_values(ascending=False)
print(nan_by_col.head(10))
m = t[t.isna().any(axis=1)]
print("rows with NaN:", len(m))
print(m.head(3))
# check overlap
k1 = set(map(tuple, base[["household_key","snapshot_day"]].drop_duplicates().values))
k2 = set(map(tuple, t[["household_key","snapshot_day"]].values))
print("in t not in base:", len(k2-k1), "in base not in t:", len(k1-k2))
