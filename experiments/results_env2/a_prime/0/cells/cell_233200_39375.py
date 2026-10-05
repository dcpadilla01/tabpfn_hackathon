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
    def wv(lo, hi):
        lo=max(lo,1); hi=min(hi,s)
        if hi<lo: return np.zeros(n)
        return cum[:,hi]-cum[:,lo-1]
    def wh(h, lo, hi):
        lo=max(lo,1); hi=min(hi,s)
        if hi<lo: return 0.0
        return cum[pos[h],hi]-cum[pos[h],lo-1]
    out = pd.DataFrame(index=idx)
    w84 = tx[tx.day >= s-83]
    if len(w84):
        hs = w84.groupby(["household_key","store_id"]).sales_value.sum().unstack(fill_value=0.0).reindex(index=idx, fill_value=0.0)
        if hs.shape[1]>0:
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
        else:
            out["st_trend_8w"] = np.nan; out["st_hh_share"] = np.nan
    else:
        out["st_trend_8w"] = np.nan; out["st_hh_share"] = np.nan
    out["st_main_spend_ratio4"] = (wv(s-27,s)+1)/(wv(s-55,s-28)+1)
    bk = tx[tx.day>=s-83].groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","first")).reset_index()
    if len(bk):
        med = bk.groupby("household_key").sp.median()
        bk["med"] = bk.household_key.map(med)
        bk["stock"] = bk.sp > 2*np.maximum(bk.med,1.0)
        out["stock_share_84"] = bk.groupby("household_key").stock.mean().reindex(idx).values
        bk["dsl"] = np.where(bk.stock, bk.d, 0)
        out["days_since_stock"] = bk.groupby("household_key").dsl.max().reindex(idx).fillna(0).values
        st_ = bk[bk.stock]
        def cyc(x):
            d = np.sort(x.values)
            return np.diff(d).mean() if len(d)>2 else np.nan
        out["stock_cycle"] = st_.groupby("household_key").d.apply(cyc).reindex(idx).values
        b28 = bk[bk.d>=s-27]
        out["bk_p90_ratio"] = b28.groupby("household_key").sp.max().reindex(idx).fillna(0).values/np.maximum(med.reindex(idx).values,1.0)
    else:
        for c in ["stock_share_84","days_since_stock","stock_cycle","bk_p90_ratio"]: out[c]=np.nan
    cc = camps[camps.start_day<=s][["campaign","description","start_day","end_day"]].rename(columns={"description":"desc"})
    ev = tgts.merge(cc, on="campaign", how="inner")
    ev = ev[(ev.start_day <= s-28) & ev.household_key.isin(idx)]
    if len(ev):
        u = np.array([(wh(r.household_key, int(r.start_day)+1, int(r.start_day)+28) - wh(r.household_key, int(r.start_day)-27, int(r.start_day)))/10.0 for r in ev.itertuples()])
        ev = ev.assign(u=u)
        out["camp_resp"] = ev.groupby("household_key").u.mean().reindex(idx).values
        out["camp_resp_n"] = ev.groupby("household_key").u.size().reindex(idx).values
        for t_ in ["TypeA","TypeB","TypeC"]:
            m = ev[ev.desc==t_].groupby("household_key").u.mean()
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
