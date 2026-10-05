import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
te_cols = [c for c in df.columns if c.startswith("te_")]
print("TE features:", te_cols)
print(m.groupby("snapshot_day")[te_cols].mean().round(2).tail(6))

# Build leak-free outcome history per household from TRAIN targets only
tr = m[m.snapshot_day<=431][["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")
tr["prev_outcomes"] = g["future_spend_4w"].cumcount()
tr["hist_mean"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
tr["hist_median"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0, drop=True)
tr["hist_last"] = g["future_spend_4w"].shift(1)
tr["hist_ewm"] = g["future_spend_4w"].apply(lambda s: s.shift(1).ewm(halflife=2).mean()).reset_index(level=0, drop=True)
tr["hist_min"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0, drop=True)
tr["hist_max"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)
tr["hist_std"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
tr["hist_n"] = tr["prev_outcomes"]
gm = tr.future_spend_4w.expanding().mean().shift(1)  # not per hh; placeholder global
tr["hist_slope"] = tr["hist_last"] - tr["hist_mean"]

t431 = tr[tr.snapshot_day==431].merge(m[m.snapshot_day==431][["household_key","te_hh_mean","te_hh_shrunk","spend_112","spend_4w_recent","nbask_4w"]], on="household_key")
y431 = t431.future_spend_4w.values
def mae(p,t): return np.mean(np.abs(np.asarray(p)-t))
print("\n== standalone predictors @431 (n=%d) ==" % len(t431))
for c in ["hist_mean","hist_median","hist_last","hist_ewm","hist_min","hist_max","hist_slope","te_hh_mean","te_hh_shrunk","spend_4w_recent"]:
    v = t431[c].fillna(t431[c].median() if t431[c].notna().any() else 0).values
    print(f"{c:14s} {mae(v,y431):8.3f}")
print("hist_n distribution @431:", t431.hist_n.describe().round(1).to_dict())

# optimal convex blend of hist stats (grid on <=403, eval 431)
tr403 = tr[tr.snapshot_day<=403]
t403 = tr403.merge(m[m.snapshot_day<=403][["household_key","snapshot_day","te_hh_mean","te_hh_shrunk"]], on=["household_key","snapshot_day"])
cands = ["hist_mean","hist_median","hist_ewm","hist_last","te_hh_mean"]
A = t403[cands].values; ya = t403.future_spend_4w.values
best=None
for w in itertools.product(*[np.linspace(0,1,6)]*len(cands)):
    w = np.array(w)
    if w.sum()==0: continue
    w = w/w.sum()
    p = A@w
    e = mae(p, ya)
    if best is None or e<best[1]: best=(w,e)
print("\nbest blend weights (fit<=403):", dict(zip(cands, best[0].round(2))), "MAE=%.3f"%best[1])
w = best[0]
p431 = t431[cands].values@w
print("blend @431 MAE:", round(mae(p431,y431),3))
