import agent_api as A, pandas as pd, numpy as np
for name in ["e001_recent_behavior.parquet","e004_temporal.parquet"]:
    t = A.load_saved(name)
    print(name, t.shape)
    print(t.columns.tolist())
    print()
tt = A.train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

TRAIN = A.snapshot_days()["train"]
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def feats_for(view, d):
    tx = view.table("transactions")
    tx = tx[tx.day <= d]
    hh = tx.household_key
    out = pd.DataFrame(index=pd.Index(sorted(hh.unique()), name="household_key"))
    def wsum(lo):
        return tx[tx.day > d-lo].groupby("household_key").sales_value.sum()
    for w in [7,28,84,365]:
        out[f"s{w}"] = wsum(w)
    g = tx.groupby("household_key")
    out["dsl"] = g.day.max().rsub(d)
    out["dsf"] = d - g.day.min() + 1
    age = (d - tx.day).astype(float)
    for hl in [28,84]:
        w = np.exp(-age*np.log(2)/hl)
        out[f"decay{hl}"] = (tx.sales_value*w).groupby(hh).sum()
    # inter-trip gap stats over last 84d
    du = tx[tx.day > d-84][["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key).agg(gap_mean="mean", gap_std="std", gap_n="count")
    out = out.join(gg)
    out["gap_cv"] = out.gap_std/out.gap_mean
    out["overdue"] = out.dsl/out.gap_mean.clip(lower=1.0)
    # weekly volatility over last 12 weeks
    w0 = (d+8)//7
    m = tx[(tx.day > d-84)]
    m = m.assign(week=(m.day+8)//7)
    m2 = m.groupby(["household_key","week"]).sales_value.sum().unstack(fill_value=0.0)
    m2 = m2.reindex(columns=range(w0-11, w0+1), fill_value=0.0)
    out["wk_cv12"] = m2.std(axis=1)/(m2.mean(axis=1)+1.0)
    # top-store share, promo share, units/trip, time of day (last 84d)
    mm = tx[tx.day > d-84]
    st = mm.groupby(["household_key","store_id"]).sales_value.sum()
    out["top_store_share"] = st.groupby(level=0).max()/st.groupby(level=0).sum().clip(lower=0.01)
    disc = mm[["retail_disc","coupon_disc","coupon_match_disc"]].clip(lower=0).sum(axis=1)
    out["promo_share"] = disc.groupby(mm.household_key).sum()/(mm.sales_value+disc).groupby(mm.household_key).sum().clip(lower=0.01)
    nb = mm.groupby("household_key").basket_id.nunique()
    out["upt84"] = mm.groupby("household_key").quantity.sum()/nb.clip(lower=1)
    out["tt_mean"] = mm.trans_time.groupby(mm.household_key).mean()
    # seasonal: same weeks last year / two years ago
    wkno = (tx.day+8)//7
    lyr = tx[wkno.isin(range(w0-51, w0-47))]
    out["hs_ly"] = lyr.groupby("household_key").sales_value.sum()
    lyr2 = tx[wkno.isin(range(w0-103, w0-99))]
    out["hs_ly2"] = lyr2.groupby("household_key").sales_value.sum()
    out["pop_ly_hh"] = lyr.sales_value.sum()/max(1, lyr.household_key.nunique())
    return out

rows=[]
for d in TRAIN:
    f = feats_for(A.snapshot(d), d)
    f["snapshot_day"]=d
    rows.append(f)
F = pd.concat(rows).reset_index()
F = F.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
print("rows", F.shape)

y = F.y; ly = np.log1p(y)
base = ["s28","s84","s365"]
Xb = np.column_stack([np.log1p(F[c].fillna(0)) for c in base])
def part(col, target):
    xc = F[col].fillna(F[col].median()).values
    X = np.column_stack([Xb, xc])
    beta,_,_,_ = np.linalg.lstsq(X, target, rcond=None)
    r_full = X@beta
    r_base = Xb@np.linalg.lstsq(Xb, target, rcond=None)[0]
    return np.corrcoef(target-r_base, r_full)[0,1]  # partial-ish

res=[]
for c in F.columns.difference(["household_key","snapshot_day","y"]):
    v = F[c]
    if v.nunique()<3: continue
    sp_raw = v.corr(y, method="spearman")
    sp_log = np.log1p(v.clip(lower=0)).corr(ly, method="spearman")
    p1 = part(c, y.values); p2 = part(c, ly.values)
    res.append((c, round(sp_raw,3), round(sp_log,3), round(p1,3), round(p2,3)))
R = pd.DataFrame(res, columns=["feat","sp_raw","sp_log","part_raw","part_log"]).sort_values("part_log", ascending=False)
print(R.to_string(index=False))
print("\nsnapshot-level: pop_ly_hh vs mean target per snapshot")
s = F.groupby("snapshot_day").agg(pop=("pop_ly_hh","first"), y=("y","mean"))
print(s.round(1)); print("corr:", s.pop.corr(s.y).round(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
def ridge_resid(X, t, lam=1.0):
    Xn = (X - X.mean(0)) / (X.std(0)+1e-9)
    Xn = np.column_stack([np.ones(len(Xn)), Xn])
    M = Xn.T@Xn + lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    beta = np.linalg.solve(M, Xn.T@t)
    return t - Xn@beta
ry = ridge_resid(Xb, F.y.values)
rly = ridge_resid(Xb, np.log1p(F.y.values))
print("base ridge R2 raw:", round(1-ry.var()/F.y.var(),4), " log:", round(1-rly.var()/np.log1p(F.y).var(),4))

cand = pd.DataFrame(index=F.index)
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
for d, sub in F.groupby("snapshot_day"):
    tx = tx_all[d]; tx = tx[tx.day<=d]
    g = tx.groupby("household_key")
    du = tx[tx.day>d-84][["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key).agg(gap_mean="mean", gap_std="std", gap_n="count")
    mm = tx[tx.day>d-84]
    nb = mm.groupby("household_key").basket_id.nunique()
    disc = mm[["retail_disc","coupon_disc","coupon_match_disc"]].clip(lower=0).sum(axis=1)
    w0=(d+8)//7
    m = tx[tx.day>d-84].assign(week=(tx[tx.day>d-84].day+8)//7)
    m2 = m.groupby(["household_key","week"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=range(w0-11,w0+1), fill_value=0.0)
    zw=[]
    for k in range(13):
        lo = d-28*(k+1); z = tx[(tx.day>lo)&(tx.day<=lo+28)].groupby("household_key").sales_value.sum()
        zw.append((z>0).astype(float))
    act = pd.concat(zw,axis=1).groupby(level=0).sum()
    c = pd.DataFrame(index=gg.index)
    c["gap_mean"]=gg.gap_mean; c["gap_std"]=gg.gap_std; c["gap_n"]=gg.gap_n
    c["gap_cv"]=gg.gap_std/gg.gap_mean
    c["overdue"]=(d-g.day.max())/gg.gap_mean.clip(lower=1.0)
    c["wk_cv12"]=m2.std(axis=1)/(m2.mean(axis=1)+1.0)
    c["decay84"]=(tx.sales_value*np.exp(-(d-tx.day)*np.log(2)/84)).groupby(tx.household_key).sum()
    st=mm.groupby(["household_key","store_id"]).sales_value.sum()
    c["top_store_share"]=st.groupby(level=0).max()/st.groupby(level=0).sum().clip(lower=0.01)
    c["upt84"]=mm.groupby("household_key").quantity.sum()/nb.clip(lower=1)
    c["unit_price"]=mm.sales_value.groupby(mm.household_key).sum()/mm.quantity.groupby(mm.household_key).sum().clip(lower=0.1)
    c["promo_share"]=disc.groupby(mm.household_key).sum()/(mm.sales_value+disc).groupby(mm.household_key).sum().clip(lower=0.01)
    c["tt_mean"]=mm.trans_time.groupby(mm.household_key).mean(); c["tt_std"]=mm.trans_time.groupby(mm.household_key).std()
    c["wknd_share"]=(mm.day%7>=5).groupby(mm.household_key).mean()
    c["active_wins_1y"]=act[0]/13.0
    b1=mm[mm.day>d-28].groupby("household_key").sales_value.sum()/mm[mm.day>d-28].basket_id.nunique().clip(lower=1)
    b0=mm[(mm.day>d-84)&(mm.day<=d-28)].groupby("household_key").sales_value.sum()/mm[(mm.day>d-84)&(mm.day<=d-28)].basket_id.nunique().clip(lower=1)
    c["basket_trend"]=b1/b0.clip(lower=0.01)
    c["snapshot_day"]=d
    cand = pd.concat([cand, c])
cand = cand.reset_index().set_index(["household_key","snapshot_day"]).loc[F.set_index(["household_key","snapshot_day"]).index]
cand["snapshot_day"]=cand.snapshot_day.values
res=[]
for col in cand.columns.difference(["snapshot_day"]):
    v=cand[col]
    if v.nunique()<3: continue
    vv=v.fillna(v.median()).values
    sp1=np.corrcoef(vv,ry)[0,1]; sp2=np.corrcoef(np.log1p(np.clip(vv,0,None)),rly)[0,1]
    res.append((col,round(sp1,3),round(sp2,3)))
print(pd.DataFrame(res,columns=["feat","resid_corr_raw","resid_corr_log"]).sort_values("resid_corr_log",ascending=False).to_string(index=False))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
def ridge_resid(X, t, lam=1.0):
    Xn = (X - X.mean(0)) / (X.std(0)+1e-9)
    Xn = np.column_stack([np.ones(len(Xn)), Xn])
    M = Xn.T@Xn + lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    beta = np.linalg.solve(M, Xn.T@t)
    return t - Xn@beta
ry = ridge_resid(Xb, F.y.values)
rly = ridge_resid(Xb, np.log1p(F.y.values))

cand = pd.DataFrame(index=F.index)
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
for d, sub in F.groupby("snapshot_day"):
    tx = tx_all[d]; tx = tx[tx.day<=d]
    g = tx.groupby("household_key")
    du = tx[tx.day>d-84][["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key).agg(gap_mean="mean", gap_std="std", gap_n="count")
    mm = tx[tx.day>d-84]
    nb = mm.groupby("household_key").basket_id.nunique()
    disc = mm[["retail_disc","coupon_disc","coupon_match_disc"]].clip(lower=0).sum(axis=1)
    w0=(d+8)//7
    m = tx[tx.day>d-84].assign(week=(tx[tx.day>d-84].day+8)//7)
    m2 = m.groupby(["household_key","week"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=range(w0-11,w0+1), fill_value=0.0)
    zw=[]
    for k in range(13):
        lo = d-28*(k+1); z = tx[(tx.day>lo)&(tx.day<=lo+28)].groupby("household_key").sales_value.sum().rename(k)
        zw.append((z>0).astype(float))
    act = pd.concat(zw,axis=1).groupby(level=0).sum()
    c = pd.DataFrame(index=gg.index)
    c["gap_mean"]=gg.gap_mean; c["gap_std"]=gg.gap_std; c["gap_n"]=gg.gap_n
    c["gap_cv"]=gg.gap_std/gg.gap_mean
    c["overdue"]=(d-g.day.max())/gg.gap_mean.clip(lower=1.0)
    c["wk_cv12"]=m2.std(axis=1)/(m2.mean(axis=1)+1.0)
    c["decay84"]=(tx.sales_value*np.exp(-(d-tx.day)*np.log(2)/84)).groupby(tx.household_key).sum()
    st=mm.groupby(["household_key","store_id"]).sales_value.sum()
    c["top_store_share"]=st.groupby(level=0).max()/st.groupby(level=0).sum().clip(lower=0.01)
    c["upt84"]=mm.groupby("household_key").quantity.sum()/nb.clip(lower=1)
    c["unit_price"]=mm.sales_value.groupby(mm.household_key).sum()/mm.quantity.groupby(mm.household_key).sum().clip(lower=0.1)
    c["promo_share"]=disc.groupby(mm.household_key).sum()/(mm.sales_value+disc).groupby(mm.household_key).sum().clip(lower=0.01)
    c["tt_mean"]=mm.trans_time.groupby(mm.household_key).mean(); c["tt_std"]=mm.trans_time.groupby(mm.household_key).std()
    c["wknd_share"]=(mm.day%7>=5).groupby(mm.household_key).mean()
    c["active_wins_1y"]=act[0]/13.0
    b1=mm[mm.day>d-28].groupby("household_key").sales_value.sum()/mm[mm.day>d-28].basket_id.nunique().clip(lower=1)
    b0=mm[(mm.day>d-84)&(mm.day<=d-28)].groupby("household_key").sales_value.sum()/mm[(mm.day>d-84)&(mm.day<=d-28)].basket_id.nunique().clip(lower=1)
    c["basket_trend"]=b1/b0.clip(lower=0.01)
    c["snapshot_day"]=d
    cand = pd.concat([cand, c])
cand = cand.reset_index().set_index(["household_key","snapshot_day"]).loc[F.set_index(["household_key","snapshot_day"]).index]
cand["snapshot_day"]=cand.snapshot_day.values
res=[]
for col in cand.columns.difference(["snapshot_day"]):
    v=cand[col]
    if v.nunique()<3: continue
    vv=v.fillna(v.median()).values
    sp1=np.corrcoef(vv,ry)[0,1]; sp2=np.corrcoef(np.log1p(np.clip(vv,0,None)),rly)[0,1]
    res.append((col,round(sp1,3),round(sp2,3)))
print(pd.DataFrame(res,columns=["feat","resid_corr_raw","resid_corr_log"]).sort_values("resid_corr_log",ascending=False).to_string(index=False))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
def ridge_resid(X, t, lam=1.0):
    Xn = (X - X.mean(0)) / (X.std(0)+1e-9)
    Xn = np.column_stack([np.ones(len(Xn)), Xn])
    M = Xn.T@Xn + lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    beta = np.linalg.solve(M, Xn.T@t)
    return t - Xn@beta
ry = ridge_resid(Xb, F.y.values)
rly = ridge_resid(Xb, np.log1p(F.y.values))

cand = pd.DataFrame(index=F.index)
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
for d, sub in F.groupby("snapshot_day"):
    tx = tx_all[d]; tx = tx[tx.day<=d]
    g = tx.groupby("household_key")
    du = tx[tx.day>d-84][["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key).agg(gap_mean="mean", gap_std="std", gap_n="count")
    mm = tx[tx.day>d-84]
    nb = mm.groupby("household_key").basket_id.nunique()
    disc = mm[["retail_disc","coupon_disc","coupon_match_disc"]].clip(lower=0).sum(axis=1)
    w0=(d+8)//7
    m = tx[tx.day>d-84].assign(week=(tx[tx.day>d-84].day+8)//7)
    m2 = m.groupby(["household_key","week"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=range(w0-11,w0+1), fill_value=0.0)
    zw=[]
    for k in range(13):
        lo = d-28*(k+1); z = tx[(tx.day>lo)&(tx.day<=lo+28)].groupby("household_key").sales_value.sum().rename(k)
        zw.append((z>0).astype(float))
    act = pd.concat(zw,axis=1).groupby(level=0).sum()
    c = pd.DataFrame(index=gg.index)
    c["gap_mean"]=gg.gap_mean; c["gap_std"]=gg.gap_std; c["gap_n"]=gg.gap_n
    c["gap_cv"]=gg.gap_std/gg.gap_mean
    c["overdue"]=(d-g.day.max())/gg.gap_mean.clip(lower=1.0)
    c["wk_cv12"]=m2.std(axis=1)/(m2.mean(axis=1)+1.0)
    c["decay84"]=(tx.sales_value*np.exp(-(d-tx.day)*np.log(2)/84)).groupby(tx.household_key).sum()
    st=mm.groupby(["household_key","store_id"]).sales_value.sum()
    c["top_store_share"]=st.groupby(level=0).max()/st.groupby(level=0).sum().clip(lower=0.01)
    c["upt84"]=mm.groupby("household_key").quantity.sum()/nb.clip(lower=1)
    c["unit_price"]=mm.sales_value.groupby(mm.household_key).sum()/mm.quantity.groupby(mm.household_key).sum().clip(lower=0.1)
    c["promo_share"]=disc.groupby(mm.household_key).sum()/(mm.sales_value+disc).groupby(mm.household_key).sum().clip(lower=0.01)
    c["tt_mean"]=mm.trans_time.groupby(mm.household_key).mean(); c["tt_std"]=mm.trans_time.groupby(mm.household_key).std()
    c["wknd_share"]=(mm.day%7>=5).groupby(mm.household_key).mean()
    c["active_wins_1y"]=act[0]/13.0
    t1 = mm[mm.day>d-28]; t0 = mm[(mm.day>d-84)&(mm.day<=d-28)]
    nb1 = t1.groupby("household_key").basket_id.nunique(); nb0 = t0.groupby("household_key").basket_id.nunique()
    b1 = t1.groupby("household_key").sales_value.sum()/nb1.clip(lower=1)
    b0 = t0.groupby("household_key").sales_value.sum()/nb0.clip(lower=1)
    c["basket_trend"]=b1/b0.clip(lower=0.01)
    c["snapshot_day"]=d
    cand = pd.concat([cand, c])
cand = cand.reset_index().set_index(["household_key","snapshot_day"]).loc[F.set_index(["household_key","snapshot_day"]).index]
cand["snapshot_day"]=cand.snapshot_day.values
res=[]
for col in cand.columns.difference(["snapshot_day"]):
    v=cand[col]
    if v.nunique()<3: continue
    vv=v.fillna(v.median()).values
    sp1=np.corrcoef(vv,ry)[0,1]; sp2=np.corrcoef(np.log1p(np.clip(vv,0,None)),rly)[0,1]
    res.append((col,round(sp1,3),round(sp2,3)))
print(pd.DataFrame(res,columns=["feat","resid_corr_raw","resid_corr_log"]).sort_values("resid_corr_log",ascending=False).to_string(index=False))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
def ridge_resid(X, t, lam=1.0):
    Xn = (X - X.mean(0)) / (X.std(0)+1e-9)
    Xn = np.column_stack([np.ones(len(Xn)), Xn])
    M = Xn.T@Xn + lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    beta = np.linalg.solve(M, Xn.T@t)
    return t - Xn@beta
ry = ridge_resid(Xb, F.y.values)
rly = ridge_resid(Xb, np.log1p(F.y.values))

frames=[]
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
for d in sorted(F.snapshot_day.unique()):
    tx = tx_all[d]; tx = tx[tx.day<=d]
    g = tx.groupby("household_key")
    du = tx[tx.day>d-84][["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key).agg(gap_mean="mean", gap_std="std", gap_n="count")
    mm = tx[tx.day>d-84]
    nb = mm.groupby("household_key").basket_id.nunique()
    disc = mm[["retail_disc","coupon_disc","coupon_match_disc"]].clip(lower=0).sum(axis=1)
    w0=(d+8)//7
    m = tx[tx.day>d-84].assign(week=(tx[tx.day>d-84].day+8)//7)
    m2 = m.groupby(["household_key","week"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=range(w0-11,w0+1), fill_value=0.0)
    zw=[]
    for k in range(13):
        lo = d-28*(k+1); z = tx[(tx.day>lo)&(tx.day<=lo+28)].groupby("household_key").sales_value.sum().rename(k)
        zw.append((z>0).astype(float))
    act = pd.concat(zw,axis=1).groupby(level=0).sum()
    c = pd.DataFrame(index=pd.Index(gg.index, name="household_key"))
    c["gap_mean"]=gg.gap_mean; c["gap_std"]=gg.gap_std; c["gap_n"]=gg.gap_n
    c["gap_cv"]=gg.gap_std/gg.gap_mean
    c["overdue"]=(d-g.day.max())/gg.gap_mean.clip(lower=1.0)
    c["wk_cv12"]=m2.std(axis=1)/(m2.mean(axis=1)+1.0)
    c["decay84"]=(tx.sales_value*np.exp(-(d-tx.day)*np.log(2)/84)).groupby(tx.household_key).sum()
    st=mm.groupby(["household_key","store_id"]).sales_value.sum()
    c["top_store_share"]=st.groupby(level=0).max()/st.groupby(level=0).sum().clip(lower=0.01)
    c["upt84"]=mm.groupby("household_key").quantity.sum()/nb.clip(lower=1)
    c["unit_price"]=mm.sales_value.groupby(mm.household_key).sum()/mm.quantity.groupby(mm.household_key).sum().clip(lower=0.1)
    c["promo_share"]=disc.groupby(mm.household_key).sum()/(mm.sales_value+disc).groupby(mm.household_key).sum().clip(lower=0.01)
    c["tt_mean"]=mm.trans_time.groupby(mm.household_key).mean(); c["tt_std"]=mm.trans_time.groupby(mm.household_key).std()
    c["wknd_share"]=(mm.day%7>=5).groupby(mm.household_key).mean()
    c["active_wins_1y"]=act[0]/13.0
    t1 = mm[mm.day>d-28]; t0 = mm[(mm.day>d-84)&(mm.day<=d-28)]
    nb1 = t1.groupby("household_key").basket_id.nunique(); nb0 = t0.groupby("household_key").basket_id.nunique()
    b1 = t1.groupby("household_key").sales_value.sum()/nb1.clip(lower=1)
    b0 = t0.groupby("household_key").sales_value.sum()/nb0.clip(lower=1)
    c["basket_trend"]=b1/b0.clip(lower=0.01)
    c["snapshot_day"]=d
    frames.append(c)
cand = pd.concat(frames).reset_index().set_index(["household_key","snapshot_day"])
cand = cand.loc[F.set_index(["household_key","snapshot_day"]).index]
cand["snapshot_day"]=cand.index.get_level_values("snapshot_day")
res=[]
for col in [c for c in cand.columns if c!="snapshot_day"]:
    v=cand[col]
    if v.nunique()<3: continue
    vv=v.fillna(v.median()).values
    sp1=np.corrcoef(vv,ry)[0,1]; sp2=np.corrcoef(np.log1p(np.clip(vv,0,None)),rly)[0,1]
    res.append((col,round(sp1,3),round(sp2,3)))
print(pd.DataFrame(res,columns=["feat","resid_corr_raw","resid_corr_log"]).sort_values("resid_corr_log",ascending=False).to_string(index=False))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
def ridge_resid(X, t, lam=1.0):
    Xn = (X - X.mean(0)) / (X.std(0)+1e-9)
    Xn = np.column_stack([np.ones(len(Xn)), Xn])
    M = Xn.T@Xn + lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    beta = np.linalg.solve(M, Xn.T@t)
    return t - Xn@beta
ry = ridge_resid(Xb, F.y.values)
rly = ridge_resid(Xb, np.log1p(F.y.values))

frames=[]
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
for d in sorted(F.snapshot_day.unique()):
    tx = tx_all[d]; tx = tx[tx.day<=d]
    g = tx.groupby("household_key")
    du = tx[tx.day>d-84][["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key).agg(gap_mean="mean", gap_std="std", gap_n="count")
    mm = tx[tx.day>d-84]
    nb = mm.groupby("household_key").basket_id.nunique()
    disc = mm[["retail_disc","coupon_disc","coupon_match_disc"]].clip(lower=0).sum(axis=1)
    w0=(d+8)//7
    m = tx[tx.day>d-84].assign(week=(tx[tx.day>d-84].day+8)//7)
    m2 = m.groupby(["household_key","week"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=range(w0-11,w0+1), fill_value=0.0)
    zw=[]
    for k in range(13):
        lo = d-28*(k+1); z = tx[(tx.day>lo)&(tx.day<=lo+28)].groupby("household_key").sales_value.sum().rename(k)
        zw.append((z>0).astype(float))
    act = pd.concat(zw,axis=1).groupby(level=0).sum()
    c = pd.DataFrame(index=pd.Index(gg.index, name="household_key"))
    c["gap_mean"]=gg.gap_mean; c["gap_std"]=gg.gap_std; c["gap_n"]=gg.gap_n
    c["gap_cv"]=gg.gap_std/gg.gap_mean
    c["overdue"]=(d-g.day.max())/gg.gap_mean.clip(lower=1.0)
    c["wk_cv12"]=m2.std(axis=1)/(m2.mean(axis=1)+1.0)
    c["decay84"]=(tx.sales_value*np.exp(-(d-tx.day)*np.log(2)/84)).groupby(tx.household_key).sum()
    st=mm.groupby(["household_key","store_id"]).sales_value.sum()
    c["top_store_share"]=st.groupby(level=0).max()/st.groupby(level=0).sum().clip(lower=0.01)
    c["upt84"]=mm.groupby("household_key").quantity.sum()/nb.clip(lower=1)
    c["unit_price"]=mm.sales_value.groupby(mm.household_key).sum()/mm.quantity.groupby(mm.household_key).sum().clip(lower=0.1)
    c["promo_share"]=disc.groupby(mm.household_key).sum()/(mm.sales_value+disc).groupby(mm.household_key).sum().clip(lower=0.01)
    c["tt_mean"]=mm.trans_time.groupby(mm.household_key).mean(); c["tt_std"]=mm.trans_time.groupby(mm.household_key).std()
    c["wknd_share"]=(mm.day%7>=5).groupby(mm.household_key).mean()
    c["active_wins_1y"]=act[0]/13.0
    t1 = mm[mm.day>d-28]; t0 = mm[(mm.day>d-84)&(mm.day<=d-28)]
    nb1 = t1.groupby("household_key").basket_id.nunique(); nb0 = t0.groupby("household_key").basket_id.nunique()
    b1 = t1.groupby("household_key").sales_value.sum()/nb1.clip(lower=1)
    b0 = t0.groupby("household_key").sales_value.sum()/nb0.clip(lower=1)
    c["basket_trend"]=b1/b0.clip(lower=0.01)
    c["snapshot_day"]=d
    frames.append(c)
cand = pd.concat(frames).reset_index()
cand = F[["household_key","snapshot_day"]].merge(cand, on=["household_key","snapshot_day"], how="left")
res=[]
for col in [c for c in cand.columns if c!="snapshot_day"]:
    v=cand[col]
    if v.nunique()<3: continue
    vv=v.fillna(v.median()).values
    sp1=np.corrcoef(vv,ry)[0,1]; sp2=np.corrcoef(np.log1p(np.clip(vv,0,None)),rly)[0,1]
    res.append((col,round(sp1,3),round(sp2,3)))
print(pd.DataFrame(res,columns=["feat","resid_corr_raw","resid_corr_log"]).sort_values("resid_corr_log",ascending=False).to_string(index=False))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
y = F.y.values
print("MAD of y (mean-abs dev from mean):", round(np.abs(y-y.mean()).mean(),2), " median:", np.median(y))

def ridge_fit(X, t, lam=1.0):
    mu=X.mean(0); sd=X.std(0)+1e-9
    Xn=(X-mu)/sd
    Xn=np.column_stack([np.ones(len(Xn)),Xn])
    M=Xn.T@Xn+lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    return np.linalg.solve(M,Xn.T@t), mu, sd
def ridge_pred(X, beta, mu, sd):
    Xn=(X-mu)/sd
    return np.column_stack([np.ones(len(Xn)),Xn])@beta

Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
ly = np.log1p(y)
beta,mu,sd = ridge_fit(Xb, ly)
pred = np.expm1(ridge_pred(Xb,beta,mu,sd))
print("E004 log-ridge in-sample MAE:", round(np.abs(pred-y).mean(),2), " R2raw:", round(1-((pred-y)**2).sum()/((y-y.mean())**2).sum(),4))

# add decay84, decay180 recomputed per snapshot
dec = np.zeros(len(F))
dec180 = np.zeros(len(F))
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
pos = {d: sub.index for d, sub in F.groupby("snapshot_day")}
for d in sorted(F.snapshot_day.unique()):
    tx = tx_all[d]; tx=tx[tx.day<=d]
    age = (d-tx.day).astype(float)
    v84 = (tx.sales_value*np.exp(-age*np.log(2)/84)).groupby(tx.household_key).sum()
    v180 = (tx.sales_value*np.exp(-age*np.log(2)/180)).groupby(tx.household_key).sum()
    idx = pos[d]
    dec[idx] = F.loc[idx,"household_key"].map(v84).fillna(0).values
    dec180[idx] = F.loc[idx,"household_key"].map(v180).fillna(0).values
X2 = np.column_stack([Xb, np.log1p(dec), np.log1p(dec180)])
beta2,mu2,sd2 = ridge_fit(X2, ly)
pred2 = np.expm1(ridge_pred(X2,beta2,mu2,sd2))
print("E004+decay84+decay180 in-sample MAE:", round(np.abs(pred2-y).mean(),2), " R2raw:", round(1-((pred2-y)**2).sum()/((y-y.mean())**2).sum(),4))

# zero structure
print("\nzero-structure:")
for c in ["spend_28","active_28"]:
    z = (y==0)
    print(c, "P(y=0|feat=0):", round(z[F[c].fillna(0)==0].mean(),3), " P(y=0|feat>0):", round(z[F[c].fillna(-1)>0].mean(),3))
# oracle-ish: predict 0 where spend_28==0 else global mean of positive
m_pos = y[y>0].mean()
pr = np.where(F.spend_28.fillna(0).values==0, 0.0, m_pos)
print("MAE predict-0-if-spend28==0-else-mean:", round(np.abs(pr-y).mean(),2))
m_pos28 = y[(y>0)&(F.spend_28.fillna(0)>0)].mean()
pr2 = np.where(F.spend_28.fillna(0).values==0, 0.0, m_pos28)
print("MAE predict-0-if-spend28==0-else-mean(pos):", round(np.abs(pr2-y).mean(),2))

# market-level trailing spend per snapshot vs mean target
rows=[]
for d in sorted(F.snapshot_day.unique()):
    tx = tx_all[d]; t = tx[(tx.day>d-84)&(tx.day<=d)]
    rows.append((d, t.sales_value.sum()/max(1,t.household_key.nunique()), t.sales_value.sum()))
M = pd.DataFrame(rows, columns=["d","mkt84","mkt_tot"]).merge(F.groupby("snapshot_day").y.mean().rename("ymean"), left_on="d", right_index=True)
print("\nmarket84 vs snapshot mean target corr:", round(M.mkt84.corr(M.ymean),3))
print(M.round(1).to_string(index=False))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def fn(view, d):
    tx = view.table("transactions")
    tx = tx[tx.day <= d]
    hh = pd.Index(list(view.households), name="household_key")
    g = tx.groupby("household_key")
    out = pd.DataFrame(index=hh)
    def wsum(lo, hi):
        return tx[(tx.day > lo) & (tx.day <= hi)].groupby("household_key").sales_value.sum()
    for w in [7,14,28,56,84,180,365]:
        out[f"spend_{w}"] = wsum(d-w, d)
    out["spend_28_prior"] = wsum(d-56, d-28)
    out["spend_84_prior"] = wsum(d-168, d-84)
    for w in [28,84]:
        out[f"baskets_{w}"] = tx[tx.day > d-w].groupby("household_key").basket_id.nunique()
    out["days_since_last"] = d - g.day.max()
    out["days_since_first"] = d - g.day.min()
    out["avg_basket_84"] = out.spend_84/out.baskets_84.clip(lower=1)
    out["trips_per_wk_84"] = out.baskets_84/12.0
    out["spend_28_ratio"] = out.spend_28/out.spend_28_prior.clip(lower=1.0)
    t84 = tx[tx.day > d-84]
    out["n_products_84"] = t84.groupby("household_key").product_id.nunique()
    out["n_stores_84"] = t84.groupby("household_key").store_id.nunique()
    out["spend_trend"] = out.spend_28 - out.spend_28_prior
    out["active_28"] = out.spend_28 > 0
    age = (d - tx.day).astype(float)
    l2 = np.log(2)
    for hl in [7,14,28,56,84,180]:
        out[f"ew_{hl}"] = (tx.sales_value*np.exp(-age*l2/hl)).groupby(tx.household_key).sum()
    for k in [336,364,392]:
        out[f"spend_lag{k}"] = wsum(d-k-28, d-k)
    out["longrun_wk"] = out.spend_365/52.0
    out["ratio28_lr"] = out.spend_28/(out.longrun_wk*4).clip(lower=0.01)
    out["ratio84_lr"] = out.spend_84/(out.longrun_wk*12).clip(lower=0.01)
    b84 = t84.groupby(["household_key","basket_id"]).sales_value.sum()
    bb = b84.groupby(level=0)
    out["basket_max_84"] = bb.max(); out["basket_std_84"] = bb.std(); out["basket_med_84"] = bb.median()
    out["active_days_28"] = tx[tx.day > d-28].groupby("household_key").day.nunique()
    # NEW: trip-spacing volatility over 84d
    du = t84[["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key)
    out["gap_cv"] = gg.std()/gg.mean()
    return out

path = A.save_table(A.build_features(fn), "e005_decay_gapcv.parquet")
print(path)
