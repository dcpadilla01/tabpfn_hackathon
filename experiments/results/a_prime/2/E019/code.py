import agent_api, pandas as pd, numpy as np
print("SNAPDAYS", agent_api.snapshot_days())
tt = agent_api.train_targets()
print("targets shape", tt.shape)
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','std','median']).round(1))

def L(nm):
    err=None
    for cand in (nm, nm+".parquet"):
        try:
            return agent_api.load_saved(cand)
        except Exception as e:
            err=e
    print("FAIL", nm, repr(err)[:120]); return None

names=["e009_ewma_longlags","e011_pruned","e013_full","e013_new_feats","e014_full_pool","e015_hh_identity","e016_dec_predictors","e017_hazard_ext","e017_hazard_only","e017_season_hazard","e017_dept_season","e018_cycle_projection","e008_main_plus_marketing","temporal_structure","basket_tenure","recency_agg","dept_mix_recency","weekly_history","marketing_exposure","product_habits","behavioral_candidates","churn_seasonality","e012_robust"]
tabs={}
for nm in names:
    d=L(nm)
    if d is not None:
        tabs[nm]=d
        ok=('household_key' in d.columns) and ('snapshot_day' in d.columns)
        print(f"{nm:26s} {str(d.shape):14s} keys_ok={ok}")

e009=tabs["e009_ewma_longlags"]
cols=list(e009.columns)
print("\nE009 ncols", len(cols))
print("E009 dtypes:", e009.dtypes.value_counts().to_dict())
print("object cols:", [c for c in cols if e009[c].dtype==object])
print("COLS:", "|".join(map(str,cols)))
b=agent_api.baseline_features()
print("baseline", b.shape)


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def L(nm):
    for c in (nm, nm+".parquet"):
        try: return agent_api.load_saved(c)
        except Exception: pass
    raise RuntimeError(nm)

base = L("e011_pruned"); haz = L("e017_hazard_only"); dec = L("e016_dec_predictors")
bcols = set(base.columns)
haz_new = [c for c in haz.columns if c not in bcols and c not in ("household_key","snapshot_day")]
dec_new = [c for c in dec.columns if c not in bcols and c not in ("household_key","snapshot_day")]
print("haz_new:", haz_new)
print("dec_new:", dec_new)

m = base.merge(haz[["household_key","snapshot_day"]+haz_new], on=["household_key","snapshot_day"], how="left")
m = m.merge(dec[["household_key","snapshot_day"]+dec_new], on=["household_key","snapshot_day"], how="left")
# drop the 7 redundant lg2_* cols added in the previous build
m = m.drop(columns=[c for c in m.columns if c.startswith("lg2_")])
feats = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("final", m.shape, "n_feats", len(feats))
print("dups:", m.columns[m.columns.duplicated()].tolist())
p = agent_api.save_table(m, "e019_curated2")
print("saved", p)


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def L(nm):
    for c in (nm, nm+".parquet"):
        try: return agent_api.load_saved(c)
        except Exception: pass
    raise RuntimeError(nm)

base = L("e011_pruned"); haz = L("e017_hazard_only"); dec = L("e016_dec_predictors")
bcols = set(base.columns)
haz_new = [c for c in haz.columns if c not in bcols and c not in ("household_key","snapshot_day")]
dec_new = [c for c in dec.columns if c not in bcols and c not in ("household_key","snapshot_day")]
drop_junk = {"index","d28_ ","sh112_ "} | {c for c in haz_new+dec_new if c.startswith(("tlag_16","tlag_17","tlag_18","d28_ELECT","d28_HOUSEWARES","d28_MEAT-WHSE","d28_PHARMACY","d28_POSTAL","d28_VIDEO","sh112_ELECT","sh112_MEAT-WHSE","sh112_PHARMACY","sh112_VIDEO"))}
haz_new = [c for c in haz_new if c not in drop_junk]
dec_new = [c for c in dec_new if c not in drop_junk]
print("haz_new:", haz_new); print("dec_new:", dec_new)

m = base.merge(haz[["household_key","snapshot_day"]+haz_new], on=["household_key","snapshot_day"], how="left")
m = m.merge(dec[["household_key","snapshot_day"]+dec_new], on=["household_key","snapshot_day"], how="left")
feats = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("final", m.shape, "n_feats", len(feats))
p = agent_api.save_table(m, "e019_curated3")
print("saved", p)
