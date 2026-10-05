import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
tr = m[m.snapshot_day<=431][["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
for hl in [1, 1.5, 2, 3, 4, 6]:
    tr[f"ewm{hl}"] = g.apply(lambda s: s.shift(1).ewm(halflife=hl).mean()).reset_index(level=0, drop=True)
tr["last1"] = g.shift(1)
tr["last2m"] = g.apply(lambda s: s.shift(1).rolling(2).mean()).reset_index(level=0, drop=True)
tr["last3m"] = g.apply(lambda s: s.shift(1).rolling(3).mean()).reset_index(level=0, drop=True)
tr["hist_mean"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
tr["hist_median"] = g.apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0, drop=True)
tr["hist_min"] = g.apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0, drop=True)
tr["hist_max"] = g.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)
tr["hist_std"] = g.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
tr["hist_slope"] = g.apply(lambda s: s.shift(1).diff()).reset_index(level=0, drop=True)

def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))
print("== standalone @431 ==")
for c in [c for c in tr.columns if c.startswith("ewm") or c in ("last1","last2m","last3m","hist_mean","hist_median","hist_min","hist_max","hist_std","hist_slope")]:
    t431 = tr[tr.snapshot_day==431]
    print(f"{c:10s} {mae(t431[c].values, t431.future_spend_4w.values):8.3f}")

t431 = tr[tr.snapshot_day==431].merge(m[m.snapshot_day==431][["household_key","te_hh_mean","te_hh_shrunk","spend_4w_recent","nbask_4w","spend_112"]], on="household_key")
t403 = tr[tr.snapshot_day<=403].merge(m[m.snapshot_day<=403][["household_key","snapshot_day","te_hh_mean","te_hh_shrunk","spend_4w_recent","nbask_4w","spend_112"]], on=["household_key","snapshot_day"])
cands = ["ewm1","ewm1.5","ewm2","ewm3","ewm4","ewm6","last1","last2m","last3m","hist_mean","hist_median","hist_min","hist_max","hist_std","hist_slope","te_hh_mean","te_hh_shrunk","spend_4w_recent"]
A = t403[cands].values; ya = t403.future_spend_4w.values
rng = np.random.default_rng(0)
best_overall = (None, 1e9, None)
for size in range(1,6):
    for combo in itertools.combinations(range(len(cands)), size):
        Ac = A[:, list(combo)]
        wbest=None; ebest=1e9
        for it in range(200):
            w = rng.dirichlet(np.ones(size)*2) if it>0 else np.ones(size)/size
            e = mae(Ac@w, ya)
            if e<ebest: ebest, wbest = e, w.copy()
        if ebest < best_overall[1]:
            best_overall = ([cands[i] for i in combo], ebest, wbest)
names, e, w = best_overall
print("\nbest blend fit<=403:", names, "MAE=%.3f"%e, np.round(w,2))
print("same blend @431 MAE:", round(mae(t431[names].values@w, t431.future_spend_4w.values),3))
for pair in [("ewm2","te_hh_mean"),("ewm1.5","te_hh_mean"),("ewm2","spend_4w_recent"),("ewm1.5","ewm2","te_hh_mean"),("ewm1","ewm2","te_hh_mean")]:
    A2 = t403[list(pair)].values; wbest=None; ebest=1e9
    for it in range(400):
        w2 = rng.dirichlet(np.ones(len(pair))*2)
        e2 = mae(A2@w2, ya)
        if e2<ebest: ebest,wbest=e2,w2
    print(pair, "fit<=403 MAE=%.3f"%ebest, "@431:", round(mae(t431[list(pair)].values@wbest, t431.future_spend_4w.values),3), np.round(wbest,2))
