
import pandas as pd, numpy as np, time

def add_feats(view, s):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.day <= s]
    w = (s + 8) // 7
    out = pd.DataFrame(index=hh)

    # weekly spend series (last 52 weeks, selected offsets)
    tw = tx[(tx.week_no >= w-52) & (tx.week_no <= w-1)]
    pv = tw.pivot_table(index="household_key", columns="week_no", values="sales_value", aggfunc="sum")
    for k in [1,2,3,4,5,6,7,8,13,26,39,52]:
        col = w - k
        out[f"wk_{k}"] = pv[col].reindex(hh).fillna(0.0) if col in pv.columns else 0.0
    aw = (pv > 0).sum(axis=1).reindex(hh).fillna(0)
    out["wk_active_8"] = np.minimum(aw, 8)

    # shifted 28d windows (previous periods, same length)
    for nm, lo, hi in [("lag28_56", s-55, s-28), ("lag56_84", s-83, s-56), ("lag84_112", s-111, s-84),
                       ("sp28_yag", s-391, s-364), ("sp84_yag", s-447, s-364)]:
        m = tx[(tx.day >= lo) & (tx.day <= hi)]
        out[nm] = m.groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)

    # day-of-week spend shares (84d); dow = day % 7
    t84 = tx[tx.day > s-84]
    dow = t84.assign(dow=t84.day % 7).pivot_table(index="household_key", columns="dow",
                                                  values="sales_value", aggfunc="sum").reindex(hh).fillna(0.0)
    tot = dow.sum(axis=1)
    for d in range(7):
        out[f"dow_{d}"] = (dow[d] / tot.replace(0, np.nan)).fillna(0.0)
    # weekend share
    out["wkend_share"] = (dow.get(5, 0) + dow.get(6, 0)) / tot.replace(0, np.nan)
    out["wkend_share"] = out["wkend_share"].fillna(0.0)

    # store concentration (84d)
    st = t84.groupby(["household_key", "store_id"]).sales_value.sum()
    out["topstore_share_84"] = (st.groupby(level=0).max() / st.groupby(level=0).sum()).reindex(hh).fillna(0.0)
    # distinct commodities (84d)
    out["ncommod_84"] = t84.groupby("household_key").commodity_desc.nunique().reindex(hh).fillna(0) \
        if "commodity_desc" in t84.columns else 0.0
    return out

def fn(view, s):
    return add_feats(view, s)

t0 = time.time()
F2 = build_features(fn)
print("built", F2.shape, "%.0fs" % (time.time()-t0))
print(F2.head(3).T.round(2))
save_table(F2, "f_weekly.parquet")
