import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

def yearago_fn(view, snapshot_day):
    hh = view.households
    hh = pd.Index(hh)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx.day
    def win(lo, hi):
        m = (d > lo) & (d <= hi)
        t = tx[m]
        g = t.groupby("household_key")
        return g.sales_value.sum(), g.basket_id.nunique()
    s1,b1 = win(snapshot_day-363, snapshot_day-336)   # year-ago matching 4w
    s2,b2 = win(snapshot_day-391, snapshot_day-336)   # year-ago 8w
    s3,b3 = win(snapshot_day-727, snapshot_day-700)   # two-years-ago 4w
    out = pd.DataFrame({
        "spend_y1_4w": s1, "baskets_y1_4w": b1,
        "spend_y1_8w": s2, "spend_y2_4w": s3,
    }).reindex(hh).fillna(0.0)
    out["active_y1"] = (out.spend_y1_4w > 0).astype(float)
    return out

t0=time.time()
X = A.build_features(yearago_fn)
print("built", X.shape, "%.0fs" % (time.time()-t0))
print(X.head(3))

feats3 = A.load_saved("feats_v3.parquet")
f5 = feats3.merge(X.drop(columns=["household_key"], errors="ignore"), left_on=["household_key","snapshot_day"],
                  right_index=True if X.index.name=="household_key" else ["snapshot_day", X.index.name or X.columns[0]])
print("merge check", f5.shape)
