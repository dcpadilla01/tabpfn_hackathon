import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, save_table

def camp_feats(view, snapshot_day):
    hh = sorted(set(int(x) for x in view.households))
    s = int(snapshot_day)
    idx = pd.Index(hh, name="household_key")
    try:
        camp = view.table("campaigns"); ct = view.table("campaign_targets")
        m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
        st = pd.to_numeric(m["start_day"], errors="coerce").astype(float).values
        en = pd.to_numeric(m["end_day"], errors="coerce").astype(float).values
        m = m.assign(st=st, en=en)
        m["hh"] = m["household_key"].astype(int).values
        m = m[m["hh"].isin(set(hh))]
        m["active"]   = (m.en >= s).astype(float)
        m["start28"]  = ((m.st >= s-27) & (m.st <= s)).astype(float)
        m["endfut"]   = ((m.en > s) & (m.en <= s+28)).astype(float)
        m["futdays"]  = np.clip(np.minimum(m.en, s+28) - np.maximum(m.st, s+1) + 1, 0, None)
        m["desc"] = m["description"].astype(str)
        g = m.groupby("hh")
        out = pd.DataFrame(index=idx)
        for c in ["active","start28","endfut","futdays"]:
            out["cmp_"+c] = g[c].sum().reindex(idx).fillna(0.0).astype(float)
        out["cmp_laststart"] = (s - g["st"].max()).reindex(idx).fillna(9999.0).astype(float)
        for t in ["TypeA","TypeB","TypeC"]:
            mt = m[m["desc"]==t]
            out["cmp_act_"+t[-1]] = mt.groupby("hh")["active"].sum().reindex(idx).fillna(0.0).astype(float)
        if s >= 95:
            print("camp feats ok", out.shape, out.mean().round(2).to_dict())
        return out
    except Exception as e:
        print("camp fallback:", type(e).__name__, str(e)[:100])
        return pd.DataFrame(0.0, index=idx, columns=["cmp_active","cmp_start28","cmp_endfut","cmp_futdays","cmp_laststart","cmp_act_A","cmp_act_B","cmp_act_C"])

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, "%.0fs" % (time.time()-t0))
print(CF.groupby("snapshot_day").mean().round(2))
save_table(CF.reset_index(), "camp_feats")
