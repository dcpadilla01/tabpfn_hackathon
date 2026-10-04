import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

# groups
def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp_extra = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp_extra+["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]]
core = [c for c in core if not c.startswith("dm_")]
print("core(%d):"%len(core), core)
print("\ncounts: dsp%d ap%d pt%d qty%d sp%d iv%d lag%d mkt%d demo%d comp%d"%(len(dsp),len(ap),len(pt),len(qty),len(sp),len(iv),len(lag),len(mkt),len(demo),len(comp_extra)))

def ridge_eval(cols, fit_days, eval_days, lam=1.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    y = d["future_spend_4w"].values
    d_tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[d_tr].T@Xs[d_tr]+lam*np.eye(len(cols)), Xs[d_tr].T@y[d_tr])
    p = Xs@b
    m = ~d_tr
    return np.mean(np.abs(p[m]-y[m])), 1-((y[m]-p[m])**2).sum()/((y[m]-y[d_tr].mean())**2).sum()

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
print("\nGROUP DIAG (ridge, fit<=403, eval 431):")
for name, cols in [("ALL228",feats),("core",core),("core+lag",core+lag),("core+lag+decay_in_lag",core+lag),
                   ("core+lag+comp",core+lag+comp_extra),("core+lag+comp+mkt",core+lag+comp_extra+mkt),
                   ("core+lag+comp+mkt+dsp",core+lag+comp_extra+mkt+dsp),
                   ("core+lag+comp+mkt+price(qty,sp_p,pt)",core+lag+comp_extra+mkt+qty+sp+pt),
                   ("core+lag+comp+mkt+ap+iv+misc",core+lag+comp_extra+mkt+ap+iv+["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]),
                   ("core+lag+demo",core+lag+demo)]:
    mae, r2 = ridge_eval(cols, tr_days, [431])
    print(f"  {name:38s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")

# dsp correlations to pick top
r = {}
for c in dsp:
    x = pd.to_numeric(df[c], errors="coerce").fillna(0).values
    r[c] = abs(np.corrcoef(x, df["future_spend_4w"].values)[0,1])
print("\ntop dsp:", sorted(r.items(), key=lambda kv:-kv[1])[:8])

# engineered features check
sp28 = pd.to_numeric(df["spend28"],errors="coerce").fillna(0)
sp56 = pd.to_numeric(df["spend56"],errors="coerce").fillna(0)
sp112 = pd.to_numeric(df["spend112"],errors="coerce").fillna(0)
rec = pd.to_numeric(df["recency"],errors="coerce").fillna(999)
tr28 = pd.to_numeric(df["trips28"],errors="coerce").fillna(0)
cons28 = sp28; cons56 = sp56/2; cons112 = sp112/4
w = np.clip(tr28/4.0, 0, 1)
blend = w*cons28 + (1-w)*cons112
dorm = cons112*np.exp(-rec/21.0)
y = df["future_spend_4w"].values
print("\neng feats corr with y: blend=%.3f dorm=%.3f cons112=%.3f (spend112=%.3f)"%(
    np.corrcoef(blend,y)[0,1], np.corrcoef(dorm,y)[0,1], np.corrcoef(cons112,y)[0,1], np.corrcoef(sp112,y)[0,1]))
