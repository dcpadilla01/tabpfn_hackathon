import agent_api, pandas as pd, numpy as np, time

f3 = agent_api.load_saved("feats_v3.parquet")
obj = [c for c in f3.columns if f3[c].dtype == object]
print("object cols:", obj)
print(f3[[c for c in f3.columns if c.startswith("dem")]].head(3))

LAGS = [84, 112, 168, 252, 308, 364]

def fn(view, snapshot_day):
    d = snapshot_day
    hh = view.households
    if hasattr(hh, "columns"):
        if "household_key" in list(hh.columns):
            keys = pd.Index(hh["household_key"].astype(int))
        else:
            keys = pd.Index(hh.index.astype(int))
    else:
        keys = pd.Index([int(k) for k in hh])
    if d == 95:
        print("hh type:", type(hh), "n_keys:", len(keys))
    tr = view.table("transactions")
    out = pd.DataFrame(index=keys)
    for L in LAGS:
        if d >= L:
            lo, hi = d + 1 - L, d + 28 - L   # same 4-week window, L days earlier
            m = tr[(tr.day >= lo) & (tr.day <= hi)]
            g = m.groupby("household_key").agg(sp=("sales_value", "sum"), bk=("basket_id", "nunique"))
            out[f"lag{L}_spend"] = g["sp"].reindex(keys).fillna(0.0)
            out[f"lag{L}_bask"] = g["bk"].reindex(keys).fillna(0.0)
        else:
            out[f"lag{L}_spend"] = np.nan
            out[f"lag{L}_bask"] = np.nan
    if d >= 391:  # 8-week window one year ago
        m = tr[(tr.day >= d - 390) & (tr.day <= d - 308)]
        g = m.groupby("household_key")["sales_value"].sum()
        out["lag364_8w"] = g.reindex(keys).fillna(0.0)
    else:
        out["lag364_8w"] = np.nan
    return out

t0 = time.time()
fn_out = agent_api.build_features(fn)
print("build time", round(time.time()-t0,1), "shape:", fn_out.shape)
fn_out = fn_out.reset_index(drop=True) if fn_out.index.name else fn_out
print(fn_out.head(3))
# coverage by snapshot
cov = fn_out.groupby("snapshot_day")[["lag84_spend","lag168_spend","lag252_spend","lag308_spend","lag364_spend","lag364_8w"]].apply(lambda g: g.notna().mean().round(2))
print(cov)
agent_api.save_table(fn_out, "feats_seasonal.parquet")
print("saved")
