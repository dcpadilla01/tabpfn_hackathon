import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
e011 = A.load_saved("e011_price.parquet")
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
dm = [c for c in feats if c.startswith("dm_")]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
keep = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+dm+misc]
keep116 = keep + lag + comp
assert len(keep116)==116, len(keep116)
prune = e011[["household_key","snapshot_day"]+keep116]
A.save_table(prune, "e017_prune.parquet")
print("saved e017_prune:", prune.shape)

# probe interactions on prune116 base
def huber_eval(df, cols, fit_days, eval_days, lam=1.0, delta=30.0, n_iter=25):
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
    return np.abs(Xs[m]@b - y[m]).mean()

df["sp112"] = pd.to_numeric(df.spend112,errors="coerce").fillna(0)
df["sp28"]  = pd.to_numeric(df.spend28,errors="coerce").fillna(0)
df["rec"]   = pd.to_numeric(df.recency,errors="coerce").fillna(999)
df["trend"] = pd.to_numeric(df.trend_112_364,errors="coerce").fillna(1).clip(-2,3)
df["oly"]   = pd.to_numeric(df.own_ly_spend4w,errors="coerce").fillna(0)
df["i_sp112_trend"] = df.sp112*df.trend
df["i_sp112_rec"]   = df.sp112*df.rec/100.0
df["i_sp28_rec"]    = df.sp28*df.rec/100.0
df["i_sp112_oly"]   = df.sp112*df.oly.clip(0,500)
df["i_sp28_oly"]    = df.sp28*df.oly.clip(0,500)
df["i_sp112_sq"]    = df.sp112**2/1000.0
df["i_sp28_sq"]     = df.sp28**2/1000.0
ic = ["i_sp112_trend","i_sp112_rec","i_sp28_rec","i_sp112_oly","i_sp28_oly","i_sp112_sq","i_sp28_sq"]
b0_431 = huber_eval(df, keep116, tr_days, [431]); b0_403 = huber_eval(df, keep116, tr_days[:-1], [403])
print(f"\nprune116 base: 431={b0_431:.3f} 403={b0_403:.3f}")
for c in ic:
    m431 = huber_eval(df, keep116+[c], tr_days, [431]); m403 = huber_eval(df, keep116+[c], tr_days[:-1], [403])
    print(f"  +{c:16s} 431={m431:.3f}({m431-b0_431:+.3f})  403={m403:.3f}({m403-b0_403:+.3f})")
all4 = huber_eval(df, keep116+ic, tr_days, [431]); all4b = huber_eval(df, keep116+ic, tr_days[:-1], [403])
print(f"  +all 7          431={all4:.3f}({all4-b0_431:+.3f})  403={all4b:.3f}({all4b-b0_403:+.3f})")
