import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()

g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh)
macro_w = macro_w[macro_w.index <= 64]

train_days = A.snapshot_days()["train"]
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index)
    W = piv.values
    avail = np.array([ (s - 28*k - 27) >= fd.values for k in range(26) ])
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W*wgt).sum(1)/wsum
    p26 = decay(4,26); p13h2 = decay(2,13)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    dd = t.groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(max(1,s-181), s+1)); full.loc[dd.index] = dd
    ew28 = float(full.ewm(halflife=28).mean().iloc[-1]*28)
    win = t[t.day > s-28]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl) else np.nan
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean()
    macro_all = mw_t.mean()
    df = pd.DataFrame({"tenure": s - fd, "p26": p26, "p13h2": p13h2, "usual_act": usual_act,
                       "ewma28": ew28, "macro_t28": macro_t28, "macro_fly": macro_fly,
                       "macro_lvl": macro_lvl, "macro_all": macro_all})
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["macro_rel"] = M.macro_lvl / M.macro_all
M["macro_fut_rel"] = M.macro_fly / M.macro_t28
M["day_idx"] = M.snapshot_day
M["e6_rvu"] = M.e6 * M.rvu.fillna(1).clip(lower=0, upper=3)
M["e6_ly"] = M.e6 * M.ly_ratio.fillna(1).clip(lower=0, upper=3)
M["e6_p26"] = M.e6 * M.p26.fillna(0)
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c: (c-c.mean())/(c.std()+1e-9)).mean(axis=1)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
M["e6_sq"] = (M.e6/100)**2

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    pred = Xho_s@b
    return np.abs(pred - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
print("LOSO(1) E008 feats:", round(loso(M, E8),3), " LOSO(2):", round(loso(M, E8, hold_last=2),3))
newf = ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_lvl","macro_all",
        "macro_rel","macro_fut_rel","day_idx","e6_rvu","e6_ly","e6_p26","consensus","sqrt_e6","e6_sq"]
for f in newf:
    print(f"LOSO E008+{f:13s}: {loso(M, E8+[f]):.3f} / {loso(M, E8+[f], hold_last=2):.3f}")