import numpy as np, pandas as pd
from agent_api import build_features, save_table

def fn(view, s):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    rel_all = (s - tx["day"]).values.astype(int)
    f = {}
    # --- phase-aligned weekly spend, 3 trailing 28d cycles ---
    m = rel_all <= 83
    t = tx.loc[m, ["household_key", "sales_value"]].copy()
    r = rel_all[m]
    t["k"] = r // 28 + 1
    t["j"] = 3 - (r % 28) // 7
    if len(t):
        pv = t.pivot_table(index="household_key", columns=["k", "j"], values="sales_value", aggfunc="sum")
    else:
        pv = pd.DataFrame()
    for kk in (1, 2, 3):
        for jj in range(4):
            col = f"c17_pw_k{kk}_w{jj+1}"
            f[col] = pv[(kk, jj)] if (kk, jj) in pv.columns else pd.Series(0.0, index=hh)
    tot1 = sum(f[f"c17_pw_k1_w{q+1}"] for q in range(4))
    for jj in range(4):
        f[f"c17_pshr_w{jj+1}"] = f[f"c17_pw_k1_w{jj+1}"] / (tot1.abs() + 1e-6)
    # --- trip gaps / cadence ---
    b = tx.groupby(["household_key", "basket_id"], as_index=False)["day"].max()
    b112 = b[b["day"] >= s - 111].sort_values(["household_key", "day"])
    if len(b112):
        b112 = b112.assign(gap=b112.groupby("household_key")["day"].diff())
        gs = b112.dropna(subset=["gap"]).groupby("household_key")["gap"]
        f["c17_gap_mean112"] = gs.mean(); f["c17_gap_med112"] = gs.median()
        f["c17_gap_std112"] = gs.std(); f["c17_gap_max112"] = gs.max()
        f["c17_gap_big112"] = gs.apply(lambda x: float((x >= 10).mean()))
    b84 = b[b["day"] >= s - 83].sort_values(["household_key", "day"])
    if len(b84):
        b84 = b84.assign(gap=b84.groupby("household_key")["day"].diff())
        mg = b84.groupby("household_key")["gap"].max()
        last = b84.groupby("household_key")["day"].max()
        f["c17_streak84"] = np.maximum(mg.fillna(0.0), s - last)
    # --- rolling 28d window zero/low fractions over full history ---
    t2 = tx[["household_key", "sales_value"]].copy()
    t2["rel"] = rel_all
    if len(t2):
        R = int(t2["rel"].max())
        dm = t2.pivot_table(index="household_key", columns="rel", values="sales_value", aggfunc="sum")
        dm = dm.reindex(columns=range(0, R + 1)).fillna(0.0)
        S = np.hstack([np.zeros((dm.shape[0], 1)), np.cumsum(dm.values, axis=1)])
        W = S[:, 28:] - S[:, :-28]
        Ws = W[:, ::7]
        f["c17_zerofrac"] = pd.Series((Ws <= 1e-9).mean(axis=1), index=dm.index)
        f["c17_lowfrac"] = pd.Series((Ws < 10).mean(axis=1), index=dm.index)
    # --- active days, cohort-relative spend ---
    t3 = tx[["household_key", "day", "sales_value"]].copy()
    t3["rel"] = rel_all
    r28 = t3[t3["rel"] < 28]; r56 = t3[t3["rel"] < 56]
    f["c17_actdays28"] = r28.groupby("household_key")["day"].nunique()
    sp28 = r28.groupby("household_key")["sales_value"].sum()
    sp56 = r56.groupby("household_key")["sales_value"].sum()
    f["c17_rk_spend28"] = sp28.rank(pct=True)
    f["c17_rk_spend56"] = sp56.rank(pct=True)
    med = float(sp28.median())
    f["c17_rmed_spend28"] = sp28 / (med if med > 0 else 1.0)
    df = pd.DataFrame(f).reindex(hh)
    return df

df = build_features(fn)
print("built:", df.shape, "n feat:", df.shape[1]-2)
print(list(df.columns))
p = save_table(df, "e017_phase.parquet")
print("saved:", p)
