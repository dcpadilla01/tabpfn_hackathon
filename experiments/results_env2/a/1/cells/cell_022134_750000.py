import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, load_saved, save_table

def camp_feats(view, snapshot_day):
    hh = view.households
    s = int(snapshot_day)
    camp = view.table("campaigns")          # start_day <= s
    ct = view.table("campaign_targets")     # households targeted
    m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
    m = m[m.household_key.isin(set(hh))]
    g = m.groupby("household_key")
    out = pd.DataFrame(index=pd.Index(sorted(set(hh)), name="household_key"))
    n_active   = g.apply(lambda d: ((d.end_day >= s)).sum(), include_groups=False)
    n_start28  = g.apply(lambda d: ((d.start_day >= s-27) & (d.start_day <= s)).sum(), include_groups=False)
    n_endfut   = g.apply(lambda d: ((d.end_day > s) & (d.end_day <= s+28)).sum(), include_groups=False)
    camp_days  = g.apply(lambda d: np.clip(d.end_day, s+1, None).clip(upper=s+28) - np.clip(d.start_day, s+1, s+28)).clip(lower=0).groupby(level=0).sum()
    last_start = g.start_day.max()
    out["cmp_active"]   = n_active.reindex(out.index).fillna(0)
    out["cmp_start28"]  = n_start28.reindex(out.index).fillna(0)
    out["cmp_endfut"]   = n_endfut.reindex(out.index).fillna(0)
    out["cmp_fut_days"] = camp_days.reindex(out.index).fillna(0)
    out["cmp_laststart"] = (s - last_start).reindex(out.index).fillna(9999)
    ta = m[m.description=="TypeA"].groupby("household_key").apply(lambda d: (d.end_day>=s).sum(), include_groups=False)
    tb = m[m.description=="TypeB"].groupby("household_key").apply(lambda d: (d.end_day>=s).sum(), include_groups=False)
    tc = m[m.description=="TypeC"].groupby("household_key").apply(lambda d: (d.end_day>=s).sum(), include_groups=False)
    out["cmp_act_A"] = ta.reindex(out.index).fillna(0)
    out["cmp_act_B"] = tb.reindex(out.index).fillna(0)
    out["cmp_act_C"] = tc.reindex(out.index).fillna(0)
    return out

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, CF.columns.tolist(), "%.0fs" % (time.time()-t0))
print(CF.groupby("snapshot_day")[["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days"]].mean().round(2))
save_table(CF.reset_index(), "camp_feats")
