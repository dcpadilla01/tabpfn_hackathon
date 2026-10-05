import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table("transactions")
    hh = view.households
    g = tx.groupby("household_key")
    feats = pd.DataFrame(index=hh.index)
    d = snapshot_day
    for w in (28, 56, 84, 112, 168, 364):
        sub = tx[tx.day > d - w]
        feats[f"spend_{w}d"] = sub.groupby("household_key").sales_value.sum().reindex(hh.index).fillna(0.0)
        feats[f"trips_{w}d"] = sub.groupby("household_key").basket_id.nunique().reindex(hh.index).fillna(0.0)
    last = g.day.max().reindex(hh.index)
    feats["days_since_last"] = (d - last).astype(float)
    feats["recency_28"] = (feats["days_since_last"] <= 28).astype(float)
    # basket stats over 112d
    sub = tx[tx.day > d - 112]
    bb = sub.groupby(["household_key","basket_id"]).sales_value.sum()
    feats["avg_basket_112"] = bb.groupby("household_key").mean().reindex(hh.index).fillna(0.0)
    # trend: last 28d vs prior 28d
    p = tx[(tx.day > d-56) & (tx.day <= d-28)].groupby("household_key").sales_value.sum().reindex(hh.index).fillna(0.0)
    feats["trend_28"] = feats["spend_28d"] - p
    # weekly spend rate
    feats["spend_per_week_84"] = feats["spend_84d"] / 12.0
    # active weeks
    feats["active_weeks_112"] = sub.assign(wk=(sub.day//7)).groupby("household_key").wk.nunique().reindex(hh.index).fillna(0.0)
    return feats

df = agent_api.build_features(fn)
print(df.shape, df.columns.tolist())
print(df.isna().mean().head(20))
p = agent_api.save_table(df, "rfm_v1")
print(p)