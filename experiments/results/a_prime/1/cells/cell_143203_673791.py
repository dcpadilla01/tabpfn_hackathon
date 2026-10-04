import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

def fn(view, snapshot_day):
    tx = view.transactions
    need = view.households
    tx = tx[tx.household_key.isin(need)]
    g = tx.groupby("household_key")

    spend_28 = g.apply(lambda d: d.loc[d.day > snapshot_day-28, "sales_value"].sum())
    trips_28 = g.apply(lambda d: d.loc[d.day > snapshot_day-28, "basket_id"].nunique())
    spend_28 = spend_28.reindex(need).fillna(0.0)
    trips_28 = trips_28.reindex(need).fillna(0.0)

    # channel: preferred store (mode) and store switching
    store_mode = tx.groupby(["household_key","store_id"]).size().reset_index(name="n")
    idx = store_mode.groupby("household_key")["n"].idxmax()
    store_mode = store_mode.loc[idx].set_index("household_key")
    main_store = store_mode["store_id"].reindex(need)
    main_store_share = (store_mode["n"] / store_mode.groupby("household_key")["n"].transform("sum")).reindex(need)
    n_stores = g.apply(lambda d: d.loc[d.day > snapshot_day-28, "store_id"].nunique()).reindex(need).fillna(0.0)

    # departments last 56 days
    tx56 = tx[tx.day > snapshot_day-56]
    prod_dept = view.products.set_index("product_id")["department"]
    dd = tx56.join(prod_dept, on="product_id")
    dep_spend = dd.pivot_table(index="household_key", columns="department", values="sales_value", aggfunc="sum").fillna(0.0)
    dep_share = dep_spend.div(dep_spend.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    dep_share.columns = [f"dep_{c}" for c in dep_share.columns]
    dep_share = dep_share.reindex(need).fillna(0.0)

    out = pd.DataFrame({
        "spend28": spend_28,
        "trips28": trips_28,
        "n_stores28": n_stores,
        "main_store": main_store,
        "main_store_share": main_store_share,
    })
    out = out.join(dep_share)
    return out

feats = agent_api.build_features(fn)
print(feats.shape, feats.columns.tolist())
base = agent_api.load_saved("e001_txhist.parquet")
merged = base.merge(feats, on=["household_key","snapshot_day"], suffixes=("", "_new"))
print(merged.shape)
p = agent_api.save_table(merged, "e002_channel")
print(p)
