import pandas as pd, numpy as np, agent_api

def L(nm):
    for c in (nm, nm+".parquet"):
        try: return agent_api.load_saved(c)
        except Exception: pass
    raise RuntimeError(nm)

base = L("e011_pruned")            # 354 cols: E009 minus row-order artifact + 16 dead cols
haz  = L("e017_hazard_only")       # hazard/seasonality extras
dec  = L("e016_dec_predictors")    # decay-weighted target-aligned averages

print("base", base.shape, "haz", haz.shape, "dec", dec.shape)
# identify hazard cols in haz
hz_cols = [c for c in haz.columns if c not in ("household_key","snapshot_day")]
print("haz cols:", hz_cols)
dec_cols = [c for c in dec.columns if c not in ("household_key","snapshot_day")]
print("dec cols:", dec_cols)

keep_h = [c for c in hz_cols if c.startswith(("zero_rate_h","mean_h","cv_h"))][:6]
keep_d = [c for c in dec_cols if c.startswith("dec_avg")][:4]
print("keep_h", keep_h); print("keep_d", keep_d)

m = base.merge(haz[["household_key","snapshot_day"]+keep_h], on=["household_key","snapshot_day"], how="left")
m = m.merge(dec[["household_key","snapshot_day"]+keep_d], on=["household_key","snapshot_day"], how="left")

# log1p transforms of heavy-tailed spend features (recomputed here, not stored in e011)
log_pairs = {
    "lg2_ewma_2":"ewma_2", "lg2_ewma_4":"ewma_4", "lg2_tlag_mean":"tlag_mean",
    "lg2_ts_sum4":"ts_sum4", "lg2_basket_mean_84":"basket_mean_84",
    "lg2_spend_28":"spend_28", "lg2_lifetime_spend":"lifetime_spend",
}
have = [c for c in log_pairs.values() if c in m.columns]
print("log sources present:", have)
for newc, src in log_pairs.items():
    if src in m.columns:
        m[newc] = np.log1p(m[src].clip(lower=0))

feats = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("final shape", m.shape, "n_feats", len(feats))
print("dup cols:", m.columns[m.columns.duplicated()].tolist())
print("NaN cols>50%:", [c for c in feats if m[c].isna().mean()>0.5][:10])
p = agent_api.save_table(m, "e019_curated")
print("saved", p)
