
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
m = tt.merge(e8, on=["household_key","snapshot_day"])
feats = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
X = m[feats].apply(pd.to_numeric, errors="coerce")
Xf = X.fillna(X.median()).values
y = m.future_spend_4w.values
Xs = (Xf - Xf.mean(0))/(Xf.std(0)+1e-9)
lam = 1.0
b = np.linalg.solve(Xs.T@Xs + lam*np.eye(Xs.shape[1]), Xs.T@y)
res = y - Xs@b
print("ridge train MAE:", np.abs(res).mean().round(2))

rc = {c: np.corrcoef(res, Xs[:,i])[0,1] for i,c in enumerate(feats)}
print("\nE008 features by |resid corr|:")
for c,v in sorted(rc.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {c:12s} {v: .4f}")

print("\nsemantics: corr(usual13,p13)=", round(np.corrcoef(m.usual13.fillna(0), m.p13.fillna(0))[0,1],3),
      " corr(usual_med,p13)=", round(np.corrcoef(m.usual_med.fillna(0), m.p13.fillna(0))[0,1],3),
      " corr(e13,p13)=", round(np.corrcoef(m.e13.fillna(0), m.p13.fillna(0))[0,1],3))

# ---- macro weekly series ----
snap = A.snapshot(459)
tx = snap.transactions
wk = tx.day.values // 7
g = tx.assign(wk=wk).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh)
w = macro_w.index.values
pairs = [(a, a+52) for a in w if a+52 in macro_w.index]
va = np.array([macro_w[a] for a,_ in pairs]); vb = np.array([macro_w[b] for _,b in pairs])
print("\nmacro weekly per-hh spend: lag-52 corr =", round(np.corrcoef(va,vb)[0,1],3),
      " mean lvl ratio (yr2/yr1) =", round(vb.mean()/va.mean(),3))
print("macro_w by 13-week blocks (yr1 vs yr2):")
blk1 = macro_w[macro_w.index<52].groupby(macro_w[macro_w.index<52].index//13).mean()
blk2 = macro_w[(macro_w.index>=52)&(macro_w.index<104)].groupby((macro_w[(macro_w.index>=52)&(macro_w.index<104)].index-52)//13).mean()
print(" yr1:", blk1.round(1).values)
print(" yr2:", blk2.round(1).values)

# ---- candidate features on train snapshots ----
train_days = A.snapshot_days()["train"]
first_day = tx.groupby("household_key").day.min()
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    # windows
    wi = ((s - t.day - 1) // 28).clip(min=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).fillna(1)
    avail = np.array([ (s - 28*k) >= fd.values for k in range(26) ])  # window end >= first day
    W = piv.values
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W*wgt).sum(1)/wsum, wsum
    p26,_ = decay(4,26); p13,_ = decay(2,13)
    # active-window usual (hl=2, 13 win)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    # ewma daily hl=28
    tt2 = t.sort_values("day").groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(1, s+1)); full.loc[tt2.index] = tt2
    ew = full.ewm(halflife=28).mean().iloc[-1]*28
    ew = pd.Series(ew, index=piv.index).reindex(piv.index).fillna(0) if np.isscalar(ew) else ew
    # macro
    win = t[(t.day > s-28)]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl)>0 else np.nan
    df = pd.DataFrame({
        "tenure": s - fd, "p26": p26, "p13h2": p13, "usual_act": usual_act,
        "ewma28": ew if isinstance(ew, pd.Series) else pd.Series(ew, index=piv.index),
        "macro_t28": macro_t28, "macro_fly": macro_fly,
    })
    df["macro_ratio"] = df.macro_fly/df.macro_t28
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
M = m[["household_key","snapshot_day","future_spend_4w","e13","e6","e3","e_med","b75","rvu","ly_ratio","dsl","usual13","p13"]].merge(C, on=["household_key","snapshot_day"])
r = pd.Series(res, index=m.index)
M = M.merge(m[["household_key","snapshot_day"]].assign(resid=res), on=["household_key","snapshot_day"])
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c:(c-c.mean())/ (c.std()+1e-9)).mean(axis=1)
M["e6_rvu"] = M.e6 * M.rvu.clip(0,3)
M["e6_ly"] = M.e6 * M.ly_ratio.clip(0,3)
M["log_dsl"] = np.log1p(M.dsl)
M["week_fut"] = ((M.snapshot_day+15)//7) % 52
M["day_idx"] = M.snapshot_day

print("\ncandidate |corr(resid)| screen (train rows):")
for c in ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_ratio",
          "consensus","e6_rvu","e6_ly","log_dsl","week_fut","day_idx"]:
    v = M[c].astype(float).values
    ok = np.isfinite(v)
    print(f"  {c:12s} {np.corrcoef(M.resid[ok], v[ok])[0,1]: .4f}")
print("\nplain MAE of usual_act as predictor:", np.abs(M.usual_act.fillna(0)-M.future_spend_4w).mean().round(2))
