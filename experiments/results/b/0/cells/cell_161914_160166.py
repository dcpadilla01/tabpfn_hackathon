import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
e011 = A.load_saved("e011_price.parquet")
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def huber_eval(df, cols, fit_days, eval_days, lam=1.0, delta=None, n_iter=25):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = np.column_stack([np.ones(len(d)), ((X-mu)/sd).clip(-10,10).fillna(0).values])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values; m = ~tr
    Xtr, ytr = Xs[tr], y[tr]
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    for _ in range(n_iter):
        r = ytr - Xtr@b
        s = np.median(np.abs(r-np.median(r)))*1.4826 + 1e-9
        dl = delta if delta else 1.345*s
        w = np.minimum(1.0, dl/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
    p = Xs[m]@b
    return np.abs(p - y[m]).mean()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+misc and not c.startswith("dm_")]

print("HUBER probe (delta=30, lam=1; fit<=403, eval 431):")
for name, cols in [("core64",core),("core+lag",core+lag),("core+lag+comp",core+lag+comp),
                   ("E011(228)",feats)]:
    print(f"  {name:16s} n={len(cols):3d}  MAE={huber_eval(df, cols, tr_days, [431]):.3f}")

# engineered candidates
sp28 = pd.to_numeric(df.spend28,errors="coerce").fillna(0); sp56 = pd.to_numeric(df.spend56,errors="coerce").fillna(0)
sp112 = pd.to_numeric(df.spend112,errors="coerce").fillna(0); rec = pd.to_numeric(df.recency,errors="coerce").fillna(999)
tr28 = pd.to_numeric(df.trips28,errors="coerce").fillna(0); tr56 = pd.to_numeric(df.trips56,errors="coerce").fillna(0)
tr112 = pd.to_numeric(df.trips112,errors="coerce").fillna(0)
eng = pd.DataFrame(index=df.index)
eng["blend"] = np.clip(tr28/4,0,1)*sp28 + (1-np.clip(tr28/4,0,1))*(sp112/4)
eng["dorm"] = (sp112/4)*np.exp(-rec/21.0)
eng["sp28_log"] = np.log1p(sp28); eng["sp56_log"] = np.log1p(sp56); eng["sp112_log"] = np.log1p(sp112)
eng["sp364_log"] = np.log1p(pd.to_numeric(df.spend364,errors="coerce").fillna(0))
eng["sqrt28"] = np.sqrt(sp28); eng["sqrt112"] = np.sqrt(sp112)
eng["cons28_log"] = np.log1p(sp28); eng["cons112_log"] = np.log1p(sp112/4)
eng["trips28_log"] = np.log1p(tr28); eng["trips112_log"] = np.log1p(tr112)
eng["rec_sq"] = rec**2; eng["rec_cubert"] = rec**(1/3)
eng["sp28xrec"] = np.log1p(sp28)*rec
eng["sp112xrec"] = np.log1p(sp112)*rec
eng["freq28"] = tr28/28.0; eng["freq112"] = tr112/112.0
eng["spend_per_trip_log"] = np.log1p(sp28/np.maximum(tr28,1))
eng["sp28_cubert"] = sp28**(1/3); eng["sp112_cubert"] = sp112**(1/3)
engf = list(eng.columns)
df2 = pd.concat([df, eng], axis=1)
print("\nengineered (added to core+lag):")
base_cols = core+lag
print(f"  core+lag            {huber_eval(df2, base_cols, tr_days, [431]):.3f}")
print(f"  +eng({len(engf)})      {huber_eval(df2, base_cols+engf, tr_days, [431]):.3f}")
print(f"  E011+eng            {huber_eval(df2, feats+engf, tr_days, [431]):.3f}")
print(f"  eng only on E011    {huber_eval(df2, feats+engf, tr_days, [431]):.3f}")
# which eng feats help individually (added to core+lag)?
print("\nindividual eng feats (added to core+lag):")
for c in engf:
    m1 = huber_eval(df2, base_cols, tr_days, [431]); m2 = huber_eval(df2, base_cols+[c], tr_days, [431])
    print(f"  {c:20s} {m2:.3f} (d={m2-m1:+.3f})")
