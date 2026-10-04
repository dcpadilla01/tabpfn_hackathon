import pandas as pd, numpy as np
import agent_api as A

def fn(view, s):
    tx = view.transactions
    h = view.households
    if isinstance(h, pd.DataFrame):
        hh = pd.Index(h["household_key"]) if "household_key" in h.columns else pd.Index(h.index)
    else:
        hh = pd.Index(h)
    # household spend in the year-ago analogue of the future window (days s-363..s-336)
    flo, fhi = s - 363, s - 336
    tw = tx[(tx.day >= flo) & (tx.day <= fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(hh).fillna(0.0) if len(hh) else tw
    # macro weekly per-active-household spend, full weeks only (week w = days 7w..7w+6)
    wk = (tx.day // 7).astype(int)
    g = tx.assign(wk=wk).groupby("wk").agg(sales=("sales_value", "sum"), hh=("household_key", "nunique"))
    macro_w = g.sales / g.hh
    full = macro_w[macro_w.index <= (s - 6) // 7]
    macro_lvl = float(full.tail(8).mean()) if len(full) else np.nan
    macro_all = float(full.mean()) if len(full) else np.nan
    macro_rel = macro_lvl / macro_all if (macro_all and macro_all > 0) else 1.0
    # year-ago macro weeks matching the future window, relative to recent macro level
    w0, w1 = (s + 1) // 7, (s + 28) // 7
    ya = macro_w.reindex(range(w0 - 52, w1 - 52 + 1)).dropna()
    mfs = float(ya.mean() / macro_lvl) if (len(ya) >= 2 and macro_lvl and macro_lvl > 0) else 1.0
    return pd.DataFrame({"mfly": mfly, "macro_rel": macro_rel, "macro_fut_seas": mfs})

newf = A.build_features(fn)
print(newf.shape)
print(newf.groupby("snapshot_day")[["macro_rel", "macro_fut_seas"]].first().round(4).to_string())
print(newf.mfly.describe().round(2).to_string())

e8 = A.load_saved("e008_decomp2.parquet")
m = e8.merge(newf, on=["household_key", "snapshot_day"], how="left")
for c in ["macro_rel", "macro_fut_seas"]:
    m[c] = m.groupby("snapshot_day")[c].transform("first")
m["macro_rel"] = m["macro_rel"].fillna(1.0)
m["macro_fut_seas"] = m["macro_fut_seas"].fillna(1.0)
m["mfly"] = m["mfly"].fillna(0.0)
m["e6_mfs"] = m.e6 * (m.macro_fut_seas - 1.0)
m["usual13_mfs"] = m.usual13 * (m.macro_fut_seas - 1.0)
m["b75_mfs"] = m.b75 * (m.macro_fut_seas - 1.0)
print("merged:", m.shape, "| NaNs:", m.isna().sum().sum())
path = A.save_table(m, "e009_macro")
print("saved:", path)
print("new cols:", [c for c in m.columns if c not in e8.columns])