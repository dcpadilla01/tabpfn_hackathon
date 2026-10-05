
import pandas as pd, numpy as np, agent_api

# Probe 1: can load_saved be used inside build_features?
def probe_fn(view, snapshot_day):
    hh = view.households
    try:
        t = load_saved("e008_level_shape.parquet")
        sub = t[t.snapshot_day == snapshot_day].set_index("household_key")
        sub = sub.reindex(hh)
        return sub[["z_med4w_hist", "sp28"]]
    except Exception as e:
        print("ERR", type(e).__name__, e); raise

bf = agent_api.build_features(probe_fn)
print("probe shape:", bf.shape, "cols:", list(bf.columns))
print("nan frac z_med4w_hist:", bf["z_med4w_hist"].isna().mean().round(4))
print(bf.head(3))

# Probe 2: weekly total-spend spikes (holiday-like weeks)?
snap = agent_api.snapshot(459)
tx = snap.transactions
wk = tx.groupby("week_no").sales_value.sum()
print("\nweekly total spend: mean %.0f std %.0f" % (wk.mean(), wk.std()))
print("top-10 weeks:", dict(wk.nlargest(10).round(0)))
print("bottom-5 weeks:", dict(wk.nsmallest(5).round(0)))
# per-active-household weekly to normalize for panel growth
cnt = tx.groupby("week_no").household_key.nunique()
wph = wk/cnt
print("per-hh weekly: mean %.1f std %.1f; top-8:" % (wph.mean(), wph.std()), dict(wph.nlargest(8).round(1)))
