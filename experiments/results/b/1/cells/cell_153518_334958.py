import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
e7 = A.load_saved("e007_new.parquet")
e1 = A.load_saved("e001_history.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]
train_days = A.snapshot_days()["train"]

rows = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum").reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    W = piv.values
    avail = np.array([ (s - 28*k - 27) >= fd for k in range(26) ]).T
    def dec_p(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail[:,:nwin]
        wsum = wgt.sum(1); wsum[wsum==0]=1
        return (W[:,:nwin]*wgt).sum(1)/wsum
    def dec_u(hl, nwin):
        pos = (W[:,:nwin]>0) & avail[:,:nwin]
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * pos
        wsum = wgt.sum(1); wsum[wsum==0]=1
        return (np.where(pos, W[:,:nwin], 0)*wgt).sum(1)/wsum
    mw_t = macro_w[macro_w.index <= (s-6)//7]  # full weeks only
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    wf0 = (s+1)//7
    ya = macro_w.reindex(range(wf0-52, wf0-52+4)).dropna()
    mfs = (ya.mean()/macro_lvl) if len(ya)>=2 else np.nan
    flo, fhi = s-363, s-336
    tw = t[(t.day>=flo)&(t.day<=fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(piv.index).fillna(0.0).values
    # weekly activity last 8 weeks
    t8 = t[t.day > s-56]
    nact8 = t8.groupby("household_key").day.nunique().reindex(piv.index).fillna(0).values
    t7 = t[t.day > s-7].groupby("household_key").sales_value.sum().reindex(piv.index).fillna(0).values
    df = pd.DataFrame({"household_key": piv.index,
        "p1_13": dec_p(1,13), "p26_8": dec_p(8,26), "u1_13": dec_u(1,13), "u26_8": dec_u(8,26),
        "usual_act": dec_u(2,26), "nact8": nact8, "spend_7d": t7,
        "mfly": mfly, "macro_rel": macro_lvl/macro_all, "macro_fut_seas": mfs, "tenure": s-fd})
    df["snapshot_day"] = s
    rows[s] = df
C = pd.concat(rows.values(), ignore_index=True)

base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
E7 = [c for c in e7.columns if c not in ("household_key","snapshot_day")]
E1 = [c for c in e1.columns if c not in ("household_key","snapshot_day")]
E1num = [c for c in E1 if pd.api.types.is_numeric_dtype(e1[c])]
# demographics one-hot for local screen
dem = e1[["household_key","snapshot_day","has_demographics"]].copy()
for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]:
    d = pd.get_dummies(e1[c], prefix=c[:6])
    dem = pd.concat([dem, d], axis=1)
DEM = [c for c in dem.columns if c not in ("household_key","snapshot_day")]
M = M.merge(dem, on=["household_key","snapshot_day"], how="left")

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    return np.abs(Xho_s@b - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

print("SANITY (harness: E001=63.0, E007=62.8, E008=61.7):")
print("  E001:", round(loso(M, E1num),3), "/", round(loso(M, E1num,2),3))
print("  E007:", round(loso(M, E7),3), "/", round(loso(M, E7,2),3))
print("  E008:", round(loso(M, E8),3), "/", round(loso(M, E8,2),3))
print("\nCANDIDATES:")
tests = {
 "macro2": ["macro_rel","macro_fut_seas"],
 "macro2+mfly+ten": ["macro_rel","macro_fut_seas","mfly","tenure"],
 "macro2+mfly": ["macro_rel","macro_fut_seas","mfly"],
 "decay_variants": ["p1_13","p26_8","u1_13","u26_8","usual_act","nact8","spend_7d"],
 "demog": DEM,
 "macro2+decay": ["macro_rel","macro_fut_seas","p1_13","p26_8","u1_13","u26_8","usual_act","nact8","spend_7d"],
 "macro2+decay+mfly": ["macro_rel","macro_fut_seas","mfly","p1_13","p26_8","u1_13","u26_8","usual_act","nact8","spend_7d"],
}
for k,v in tests.items():
    print(f"  {k:18s}: {loso(M, E8+v):.3f} / {loso(M, E8+v,2):.3f}  (n={len(v)})")
print("\nmacro_rel by snapshot (train):"); print(M.groupby("snapshot_day").macro_rel.first().round(3).to_string())
print("macro_fut_seas by snapshot (train):"); print(M.groupby("snapshot_day").macro_fut_seas.first().round(3).to_string())