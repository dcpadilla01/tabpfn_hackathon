import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

# check dtypes / infs
bad = []
for c in feats:
    x = pd.to_numeric(df[c], errors="coerce")
    if x.isna().all(): bad.append((c,"all-nan"))
    elif np.isinf(x.values).any(): bad.append((c,"inf"))
    elif df[c].dtype == object: bad.append((c,"object"))
print("problem cols:", bad[:20], "..." if len(bad)>20 else "")

def ridge_eval(cols, fit_days, eval_days, lam=3.0, clip=10.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-clip, clip).fillna(0).values
    y = d["future_spend_4w"].values
    d_tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[d_tr].T@Xs[d_tr]+lam*np.eye(len(cols)), Xs[d_tr].T@y[d_tr])
    p = Xs@b
    m = ~d_tr
    return np.mean(np.abs(p[m]-y[m])), 1-((y[m]-p[m])**2).sum()/((y[m]-y[d_tr].mean())**2).sum()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp_extra = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp_extra+misc and not c.startswith("dm_")]

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
print("\nGROUP DIAG (ridge lam=3, fit<=403, eval 431):")
for name, cols in [("core64",core),("core+lag",core+lag),("core+lag+comp",core+lag+comp_extra),
                   ("core+lag+comp+mkt",core+lag+comp_extra+mkt),
                   ("core+lag+comp+mkt+dsp",core+lag+comp_extra+mkt+dsp),
                   ("core+lag+comp+mkt+price",core+lag+comp_extra+mkt+qty+sp+pt),
                   ("ALL228",feats)]:
    mae, r2 = ridge_eval(cols, tr_days, [431])
    print(f"  {name:26s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")

# single-snapshot eval is noisy; also do leave-one-snapshot-out style: eval on 403 too
print("\ncross-check eval on 403 (fit <=375):")
for name, cols in [("core64",core),("core+lag",core+lag),("ALL228",feats)]:
    mae, r2 = ridge_eval(cols, tr_days[:-1], [403])
    print(f"  {name:26s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")
