import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
e011 = A.load_saved("e011_price.parquet")
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def huber_eval(df, cols, fit_days, eval_days, lam=1.0, l1=0.0, delta=30.0, n_iter=25):
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
        w = np.minimum(1.0, delta/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
        if l1 > 0:  # simple l1 shrink step
            b[1:] = np.sign(b[1:]) * np.maximum(np.abs(b[1:]) - l1/ (np.diag(Xtr.T@Xw)+1e-9), 0)
    p = Xs[m]@b
    return np.abs(p - y[m]).mean()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
dm = [c for c in feats if c.startswith("dm_")]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+misc+dm]

cands = {
 "core64": core,
 "core+comp": core+comp,
 "core+lag": core+lag,
 "core+lag+comp": core+lag+comp,
 "core+lag+comp+mkt": core+lag+comp+mkt,
 "core+lag+comp+mkt+dm": core+lag+comp+mkt+dm,
 "core+lag+comp+mkt+dm+demo": core+lag+comp+mkt+dm+demo,
 "E011-dsp(184)": [c for c in feats if c not in dsp],
 "E011-junk(142)": core+lag+comp+mkt+dm+demo,
 "E011(228)": feats,
}
print("HUBER probe, lam=1, delta=30 — eval on 431 and 403:")
print(f"{'cand':28s} {'n':>4s} {'431':>8s} {'403':>8s}")
for name, cols in cands.items():
    m431 = huber_eval(df, cols, tr_days, [431])
    m403 = huber_eval(df, cols, tr_days[:-1], [403])
    print(f"{name:28s} {len(cols):4d} {m431:8.3f} {m403:8.3f}")

print("\nL1 variants (lam=1, l1 shrink) on 431:")
for name, cols in [("core64",core),("E011-junk(142)",core+lag+comp+mkt+dm+demo),("E011(228)",feats)]:
    print(f"  {name:16s} l1=0.02: {huber_eval(df, cols, tr_days, [431], l1=0.02):.3f}  l1=0.05: {huber_eval(df, cols, tr_days, [431], l1=0.05):.3f}")
