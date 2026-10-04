import pandas as pd, numpy as np

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.transactions
    hh = view.households
    f = pd.DataFrame(index=hh)

    # --- seasonal lag neighbors (LY window = [d-363, d-336] is E001's spend_ly28) ---
    ly_pre  = tx[(tx.day>=d-391)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
    ly56    = tx[(tx.day>=d-419)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
    f["ly28_pre"] = ly_pre.reindex(hh).fillna(0.0)
    f["ly_ratio_pre"] = f["ly28_pre"]/(ly56.reindex(hh).fillna(0.0)+1.0)

    # --- peer-group recent spend (demographic prior, past-only) ---
    sp28 = tx[(tx.day>=d-27)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    demo = view.demographics
    j = pd.DataFrame({"sp28": sp28}).join(demo.set_index("household_key")[["classification_1","classification_5"]])
    m1 = j.groupby("classification_1").sp28.mean(); m5 = j.groupby("classification_5").sp28.mean()
    dm = demo.set_index("household_key")
    f["peer_spend_c1"] = dm.classification_1.map(m1).reindex(hh)
    f["peer_spend_c5"] = dm.classification_5.map(m5).reindex(hh)

    # --- tenure / growth / activity flags ---
    first = tx.groupby("household_key").day.min()
    sp84 = tx[(tx.day>=d-83)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    sp7  = tx[(tx.day>=d-6)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    f["tenure"] = (d - first).reindex(hh).fillna(0.0)
    f["growth_28_84"] = sp28.reindex(hh).fillna(0.0)/(sp84.reindex(hh).fillna(0.0)+1.0)
    f["zero7"] = (sp7.reindex(hh).fillna(0.0)==0).astype(float)

    # --- empirical binned-median lookup: future spend | past-28d spend, from pseudo-snapshots ---
    parts=[]
    for s in range(d-28, 27, -28):
        w = tx[tx.day<=s]
        sp = w[w.day>=s-27].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu}).fillna(0.0))
    if parts:
        pool = pd.concat(parts)
        qs = np.unique(np.quantile(pool.sp, np.linspace(0,1,11)))
        lab = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
        med = pool.fu.groupby(lab).median()
        f["lk_median"] = pd.cut(sp28.reindex(hh).fillna(0.0), qs, labels=False, include_lowest=True).map(med).astype(float)
        f["lk_median"] = f["lk_median"].fillna(pool.fu.median())
    else:
        f["lk_median"] = np.nan
    return f

X = agent_api.build_features(fn)
print(X.shape); print(X.head(3))
path = agent_api.save_table(X, "e005_seasonal_peer")
print(path)