import agent_api as api, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
snap = api.snapshot(); tx_all = snap.transactions
SNAPS = sorted(saved.snapshot_day.unique())

def compute_batch1(tx, hhs, s, camps, tgts, reds):
    tx = tx[tx.day <= s]
    idx = pd.Index(hhs, name="household_key")
    n = len(idx)
    daily = tx.groupby(["household_key","day"]).sales_value.sum()
    piv = daily.unstack(fill_value=0.0).reindex(index=idx, columns=range(1, s+1), fill_value=0.0)
    cum = np.hstack([np.zeros((n,1)), piv.cumsum(1).values])
    pos = {h:i for i,h in enumerate(idx)}
    def wsum(lo, hi):
        lo=max(lo,1); hi=min(hi,s)
        if hi<lo: return np.zeros(n)
        return cum[:,hi]-cum[:,lo-1]
    out = pd.DataFrame(index=idx)
    w84 = tx[tx.day >= s-83]
    hs = w84.groupby(["household_key","store_id"]).sales_value.sum().unstack(fill_value=0.0).reindex(index=idx, fill_value=0.0)
    main_store = hs.values.argmax(1)
    hh_main_spend = hs.values[np.arange(n), main_store]
    store_ids = hs.columns.values[main_store]
    sw = tx.groupby(["store_id","week_no"]).sales_value.sum().unstack(fill_value=0.0)
    cur_w = (s+8)//7
    last4 = sw.reindex(columns=range(cur_w-3, cur_w+1), fill_value=0.0).sum(1)
    prev4 = sw.reindex(columns=range(cur_w-7, cur_w-3), fill_value=0.0).sum(1)
    st_trend_map = ((last4+1)/(prev4+1))
    out["st_trend_8w"] = st_trend_map.loc[store_ids].values
    st_tot = tx[tx.day>=s-83].groupby("store_id").sales_value.sum()
    out["st_hh_share"] = hh_main_spend/np.maximum(st_tot.loc[store_ids].values,1.0)
    out["st_main_spend_ratio4"] = (wsum(s-27,s)+1)/(wsum(s-55,s-28)+1)
    bk = tx[tx.day>=s-83].groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","first"))
    bk = bk.reset_index().set_index("household_key").reindex(idx)
    med_bk = bk.groupby(level=0).sp.median().reindex(idx).values
    stock = bk.sp.values > 2*np.maximum(med_bk,1.0)
    g = pd.Series(stock, index=bk.index).groupby(level=0)
    out["stock_share_84"] = g.mean().reindex(idx).values
    last_stock = bk.day.values*stock
    out["days_since_stock"] = np.where(last_stock>0, s-last_stock, s)
    sd = bk.day.values[stock]; sh = bk.index.values[stock]
    sdf = pd.DataFrame({"h":sh,"d":sd}).groupby("h").d.apply(lambda x: np.diff(np.sort(x)).mean() if len(x)>2 else np.nan)
    out["stock_cycle"] = sdf.reindex(idx).values
    b28 = bk[bk.d>=s-27]
    out["bk_p90_ratio"] = b28.groupby(level=0).sp.max().reindex(idx).fillna(0).values/np.maximum(med_bk,1.0)
    cc = camps[camps.start_day<=s][["campaign","description","start_day","end_day"]]
    ev = tgts.merge(cc, on="campaign", how="inner")
    ev = ev[(ev.start_day <= s-28) & ev.household_key.isin(idx)]
    def wh(h, lo, hi):
        lo=max(lo,1); hi=min(hi,s)
        if hi<lo: return 0.0
        return cum[pos[h],hi]-cum[pos[h],lo-1]
    if len(ev):
        u = np.array([(wh(r.household_key, int(r.start_day)+1, int(r.start_day)+28) - wh(r.household_key, int(r.start_day)-27, int(r.start_day)))/10.0 for r in ev.itertuples()])
        ev = ev.assign(u=u)
        out["camp_resp"] = ev.groupby("household_key").u.mean().reindex(idx).values
        out["camp_resp_n"] = ev.groupby("household_key").u.size().reindex(idx).values
        for t_ in ["TypeA","TypeB","TypeC"]:
            m = ev[ev.description==t_].groupby("household_key").u.mean()
            out["resp_"+t_] = m.reindex(idx).values
        last_ev = ev.groupby("household_key").start_day.max().reindex(idx)
        uu = ev.set_index(["household_key","start_day"]).u
        out["camp_resp_last"] = [uu.get((h,d), np.nan) for h,d in last_ev.items()]
    else:
        for c in ["camp_resp","camp_resp_n","resp_TypeA","resp_TypeB","resp_TypeC","camp_resp_last"]: out[c]=np.nan
    r28 = reds[(reds.day>=s-27)&(reds.household_key.isin(idx))]
    cc2 = camps[camps.start_day<=s][["campaign","end_day"]]
    r28 = r28.merge(cc2, on="campaign", how="left")
    out["n_red_active28"] = r28[r28.end_day>=s].groupby("household_key").size().reindex(idx).fillna(0).values
    return out

camps_all = snap.campaigns; tgts_all = snap.campaign_targets; reds_all = snap.coupon_redemptions
feats = {}
for s in SNAPS:
    hhs = saved[saved.snapshot_day==s].household_key.values
    feats[s] = compute_batch1(tx_all, hhs, s, camps_all, tgts_all, reds_all)
b1 = pd.concat([f.assign(snapshot_day=s) for s,f in feats.items()]).reset_index()
print(b1.shape)
api.save_table(b1, "cand_batch1")
print(b1.describe().T[["mean","std","count"]].round(2))
