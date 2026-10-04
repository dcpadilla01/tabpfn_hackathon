import agent_api as A
import pandas as pd, numpy as np

def fn(view, s):
    tx = view.transactions
    hh = pd.Index(view.households)
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby("household_key")
    # weekly windows w=1..13 (w=1: days s-6..s)
    rel = s - tx.day
    wk = (rel // 7 + 1).clip(1, 13)
    mo = (rel // 28 + 1).clip(1, 13)
    sv = tx.sales_value
    piv_w = pd.pivot_table(pd.DataFrame({"h":tx.household_key,"w":wk,"v":sv}),
                           index="h", columns="w", values="v", aggfunc="sum").reindex(hh)
    piv_m = pd.pivot_table(pd.DataFrame({"h":tx.household_key,"m":mo,"v":sv}),
                           index="h", columns="m", values="v", aggfunc="sum").reindex(hh)
    W = piv_w.reindex(columns=range(1,14)).fillna(0.0).values  # n x 13 weekly
    M = piv_m.reindex(columns=range(1,14)).fillna(0.0).values  # n x 13 28-day
    out = pd.DataFrame(index=hh)
    out["r_med13"] = np.median(W, axis=1)
    out["r_q25_13"] = np.percentile(W, 25, axis=1)
    out["r_q75_13"] = np.percentile(W, 75, axis=1)
    out["r_med4"] = np.median(W[:, :4], axis=1)
    out["r_maxmed13"] = W.max(1) / (np.median(W,1)+1.0)
    out["r_zerow13"] = (W == 0).sum(1)
    out["r_zerom13"] = (M == 0).sum(1)
    out["r_zerow4"] = (W[:, :4] == 0).sum(1)
    # lag-1 autocorr of weekly series
    ac = np.full(len(out), np.nan)
    X0, X1 = W[:, 1:-1], W[:, :-2]
    sd0, sd1 = X0.std(1), X1.std(1)
    ok = (sd0 > 1e-9) & (sd1 > 1e-9)
    ac[ok] = ((X0[ok]-X0[ok].mean(1,keepdims=True))*(X1[ok]-X1[ok].mean(1,keepdims=True))).mean(1)/(sd0[ok]*sd1[ok])
    out["r_autocorr"] = ac
    # normalized trend slope over 13 weeks
    t = np.arange(13, dtype=float); tc = t - t.mean()
    slope = (W * tc).sum(1) / (tc**2).sum()
    out["r_slope13"] = slope / (W.mean(1) + 1.0)
    # recency / gaps
    last = g.day.max().reindex(hh).fillna(0)
    out["r_dsl"] = s - last
    days_u = g.day.apply(lambda x: np.sort(x.unique())).reindex(hh)
    gaps = []
    for d in days_u.values:
        gaps.append(np.diff(d).mean() if len(d) > 1 else np.nan)
    out["r_gap_mean"] = gaps
    out["r_gap_ratio"] = out["r_dsl"] / (out["r_gap_mean"] + 1.0)
    # cross-sectional percentile ranks (drift-free level signal)
    l1 = M[:,0]; l4 = M[:, :4].sum(1); l13 = M.sum(1)
    for nm, v in [("pct_l1", l1), ("pct_l4", l4), ("pct_l13", l13)]:
        out[nm] = pd.Series(v, index=hh).rank(pct=True)
    # snapshot-level global aggregates (drift calibration)
    out["g_mean_l4"] = float(np.mean(l4)); out["g_med_l4"] = float(np.median(l4))
    out["g_mean_l1"] = float(np.mean(l1))
    # seasonality
    wky = ((s + 8) // 7) % 52
    out["k_sin1"] = np.sin(2*np.pi*wky/52.0); out["k_cos1"] = np.cos(2*np.pi*wky/52.0)
    out["k_sin2"] = np.sin(4*np.pi*wky/52.0); out["k_cos2"] = np.cos(4*np.pi*wky/52.0)
    out["s_day"] = float(s)
    return out

df = A.build_features(fn)
print("built:", df.shape)
p = A.save_table(df, "nf_robust.parquet")
print("saved:", p)

# drift diagnostics on train targets
tt = A.train_targets()
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]).round(1))
# global aggregate drift
print(df.groupby("snapshot_day")[["g_mean_l4","g_med_l4","g_mean_l1"]].first().round(1))