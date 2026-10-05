import agent_api, pandas as pd, numpy as np
t = agent_api.train_targets()
print(t.shape)
print(t.future_spend_4w.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print("zero share:", (t.future_spend_4w==0).mean())
print(t.head())

# ---- cell ----
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

# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    d = snapshot_day
    feats = pd.DataFrame(index=hh)
    for w in (28, 56, 84, 112, 168, 364):
        sub = tx[tx.day > d - w]
        feats[f"spend_{w}d"] = sub.groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
        feats[f"trips_{w}d"] = sub.groupby("household_key").basket_id.nunique().reindex(hh).fillna(0.0)
    last = tx.groupby("household_key").day.max().reindex(hh)
    feats["days_since_last"] = (d - last).astype(float)
    sub112 = tx[tx.day > d - 112]
    bb = sub112.groupby(["household_key","basket_id"]).sales_value.sum()
    feats["avg_basket_112"] = bb.groupby("household_key").mean().reindex(hh).fillna(0.0)
    p = tx[(tx.day > d-56) & (tx.day <= d-28)].groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
    feats["trend_28"] = feats["spend_28d"] - p
    feats["spend_per_week_84"] = feats["spend_84d"] / 12.0
    feats["active_weeks_112"] = sub112.assign(wk=sub112.day//7).groupby("household_key").wk.nunique().reindex(hh).fillna(0.0)
    return feats

df = agent_api.build_features(fn)
print(df.shape, len(df.columns))
print(df.isna().mean().mean())
p = agent_api.save_table(df, "rfm_v1")
print(p)