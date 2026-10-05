import agent_api as A
print(A.snapshot_days())
print(A.describe_tables())
snap = A.snapshot()
tr = snap.transactions
print(tr.shape)
print(tr.sales_value.describe())
# target distribution
tt = A.train_targets()
print(tt.future_spend_4w.describe())
print((tt.future_spend_4w==0).mean())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    hh = view.households
    tr = view.table("transactions")
    tr = tr[tr.household_key.isin(hh)]
    g = tr.groupby("household_key")
    out = pd.DataFrame(index=hh)
    d = snapshot_day
    # spend in windows
    for w in [4, 8, 12, 28, 56, 112]:
        sub = tr[(tr.day > d - w) & (tr.day <= d)]
        out[f"spend_{w}w"] = sub.groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
        out[f"nbask_{w}w"] = sub.groupby("household_key").basket_id.nunique().reindex(hh).fillna(0.0)
    out["spend_total"] = g.sales_value.sum().reindex(hh).fillna(0.0)
    out["tenure_days"] = (d - g.day.min()).reindex(hh)
    out["days_since_last"] = (d - g.day.max()).reindex(hh).fillna(999)
    out["avg_basket_12w"] = out["spend_12w"] / out["nbask_12w"].replace(0, np.nan)
    out["trend_4_8"] = out["spend_4w"] / (out["spend_8w"] / 2).replace(0, np.nan)
    out["trend_4_28"] = out["spend_4w"] / (out["spend_28w"] / 7).replace(0, np.nan)
    out["active_4w"] = (out["nbask_4w"] > 0).astype(int)
    out["day"] = float(d)
    out["week"] = (d + 8) // 7
    return out

tbl = A.build_features(fn)
print(tbl.shape)
print(tbl.head())
path = A.save_table(tbl, "e001_rfm")
print(path)
