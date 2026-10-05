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
