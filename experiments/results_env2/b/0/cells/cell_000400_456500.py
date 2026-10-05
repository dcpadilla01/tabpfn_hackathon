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
