import agent_api as api, numpy as np, pandas as pd

# --- Test: can load_saved be used inside build_features fn? ---
def fn(view, snapshot_day):
    base = api.load_saved("e013_storeprod.parquet")
    b = base[base.snapshot_day == snapshot_day].set_index("household_key")
    return b[["spend_4w_recent"]]
try:
    t = api.build_features(fn)
    saved = api.load_saved("e013_storeprod.parquet")
    m = t.merge(saved[["household_key","snapshot_day","spend_4w_recent"]], on=["household_key","snapshot_day"], suffixes=("_new","_old"))
    print("load_saved-in-fn OK; rows:", t.shape, "match:", np.allclose(m.spend_4w_recent_new, m.spend_4w_recent_old))
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:300])

# --- Understand existing lag columns ---
saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
snap = api.snapshot()
tx = snap.transactions
print("tx max day:", tx.day.max())

daily = tx.groupby(["household_key","day"]).sales_value.sum()
pivot = daily.unstack(fill_value=0.0).reindex(columns=range(1,460), fill_value=0.0)
cum = np.hstack([np.zeros((pivot.shape[0],1)), pivot.cumsum(axis=1).values])
hidx = pd.Index(pivot.index)

def win_sum(lo, hi):
    lo = max(lo,1); hi = min(hi,459)
    if hi < lo: return pd.Series(0.0, index=hidx)
    return pd.Series(cum[:,hi]-cum[:,lo-1], index=hidx)

# what window is spend_4w_lag1/2/3 ?
sub = saved[saved.snapshot_day==431].set_index("household_key")
for name,(lo,hi) in {"lag1=(s-27..s)":(431-27,431),"lag2=(s-55..s-28)":(431-55,431-28),"lag3=(s-83..s-56)":(431-83,431-56)}.items():
    v = win_sum(lo,hi).reindex(sub.index)
    print(name, "corr with saved col:", sub["spend_4w_lag1" if "lag1" in name else ("spend_4w_lag2" if "lag2" in name else "spend_4w_lag3")].corr(v))
