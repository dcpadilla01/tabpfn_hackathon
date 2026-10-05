import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, save_table

def camp_feats(view, snapshot_day):
    hh = set(view.households)
    s = int(snapshot_day)
    camp = view.table("campaigns")
    ct = view.table("campaign_targets")
    m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
    m["start_day"] = pd.to_numeric(m["start_day"]); m["end_day"] = pd.to_numeric(m["end_day"])
    m = m[m.household_key.isin(hh)]
    idx = pd.Index(sorted(hh), name="household_key")
    g = m.groupby("household_key")
    out = pd.DataFrame(index=idx)
    out["cmp_active"]   = g.apply(lambda d: float((d.end_day >= s).sum()), include_groups=False).reindex(idx).fillna(0)
    out["cmp_start28"]  = g.apply(lambda d: float(((d.start_day >= s-27) & (d.start_day <= s)).sum()), include_groups=False).reindex(idx).fillna(0)
    out["cmp_endfut"]   = g.apply(lambda d: float(((d.end_day > s) & (d.end_day <= s+28)).sum()), include_groups=False).reindex(idx).fillna(0)
    def fut_days(d):
        a = np.maximum(pd.to_numeric(d.start_day).values, s+1)
        b = np.minimum(pd.to_numeric(d.end_day).values, s+28)
        return float(np.clip(b - a + 1, 0, None).sum())
    out["cmp_fut_days"] = g.apply(fut_days, include_groups=False).reindex(idx).fillna(0)
    out["cmp_laststart"] = (s - g.start_day.max()).reindex(idx).fillna(9999).astype(float)
    for t in ["TypeA","TypeB","TypeC"]:
        mt = m[m.description==t]
        out["cmp_act_"+t[-1]] = mt.groupby("household_key").apply(lambda d: float((d.end_day>=s).sum()), include_groups=False).reindex(idx).fillna(0)
    return out

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, "%.0fs" % (time.time()-t0))
print(CF.groupby("snapshot_day")[["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days","cmp_act_A","cmp_act_B","cmp_act_C"]].mean().round(2))
save_table(CF.reset_index(), "camp_feats")
