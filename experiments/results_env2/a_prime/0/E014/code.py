import agent_api as api
df = api.load_saved("e013_storeprod.parquet")
print(df.shape)
cols = list(df.columns)
print(len(cols))
for c in cols: print(c)

# ---- cell ----
import agent_api as api, numpy as np, pandas as pd

# --- Test: can load_saved be used inside build_features fn? ---
def fn(view, snapshot_day):
    base = api.load_saved("e013_storeprod.parquet")
    b = base[base.snapshot_day == snapshot_day].set_index("household_key")
    return b[["spend_4w_recent"]]
try:
    t = api.build_features(fn)
    saved = api.load_saved("e013_storeprod.parquet")
    m = t.merge(saved[["household_key","snapshot_day","spend_4w_recent"]], on=["household_key","snapshot_day"], suffixes=("_new","_old"))
    print("load_saved-in-fn OK; rows:", t.shape, "match:", np.allclose(m.spend_4w_recent_new, m.spend_4w_recent_old))
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:300])

# --- Understand existing lag columns ---
saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
snap = api.snapshot()
tx = snap.transactions
print("tx max day:", tx.day.max())

daily = tx.groupby(["household_key","day"]).sales_value.sum()
pivot = daily.unstack(fill_value=0.0).reindex(columns=range(1,460), fill_value=0.0)
cum = np.hstack([np.zeros((pivot.shape[0],1)), pivot.cumsum(axis=1).values])
hidx = pd.Index(pivot.index)

def win_sum(lo, hi):
    lo = max(lo,1); hi = min(hi,459)
    if hi < lo: return pd.Series(0.0, index=hidx)
    return pd.Series(cum[:,hi]-cum[:,lo-1], index=hidx)

# what window is spend_4w_lag1/2/3 ?
sub = saved[saved.snapshot_day==431].set_index("household_key")
for name,(lo,hi) in {"lag1=(s-27..s)":(431-27,431),"lag2=(s-55..s-28)":(431-55,431-28),"lag3=(s-83..s-56)":(431-83,431-56)}.items():
    v = win_sum(lo,hi).reindex(sub.index)
    print(name, "corr with saved col:", sub["spend_4w_lag1" if "lag1" in name else ("spend_4w_lag2" if "lag2" in name else "spend_4w_lag3")].corr(v))


# ---- cell ----
import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")  # closure variable

def fn(view, snapshot_day):
    info = {}
    tx = view.table("transactions")
    info["view_day"] = int(view.day)
    info["tx_max_day"] = int(tx.day.max()) if len(tx) else -1
    info["n_hh"] = len(view.households)
    b = saved[saved.snapshot_day == snapshot_day].set_index("household_key")
    info["closure_rows"] = len(b)
    out = b[["spend_4w_recent"]].copy()
    out["view_day_probe"] = float(view.day)
    print(snapshot_day, info)
    return out

t = api.build_features(fn)
print(t.shape, t.columns.tolist())


# ---- cell ----
import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
y = tt.future_spend_4w.values
print("target: mean %.1f med %.1f std %.1f zero-share %.3f" % (y.mean(), np.median(y), y.std(), (y==0).mean()))
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median",lambda x:(x==0).mean()]))

# ridge proxy on E013 numeric features
df = saved.merge(tt, on=["household_key","snapshot_day"])
num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number]).columns.tolist()
print("n numeric feats:", len(num))
X = df[num].values.astype(float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
mu, sd = X[df.snapshot_day<=431].mean(0), X[df.snapshot_day<=431].std(0)+1e-9
Xs = (X-mu)/sd
tr = (df.snapshot_day<=431).values; va = ~tr
def ridge(lam):
    A = Xs[tr]; b = y[tr]
    I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@b)
    pred = Xs@w
    return pred
best=None
for lam in [1,10,100,300,1000,3000]:
    p = ridge(lam)
    mae = np.abs(p[va]-y[va]).mean()
    if best is None or mae<best[1]: best=(lam,mae)
print("ridge proxy best:", best)
p = ridge(best[0])
res = y-p
print("resid: mean %.2f std %.2f | corr resid vs lag1 %.3f vs te2_ewm %.3f vs te_hh_mean %.3f" % (
    res.mean(), res.std(), np.corrcoef(res, df.spend_4w_lag1)[0,1], np.corrcoef(res, df.te2_ewm)[0,1], np.corrcoef(res, df.te_hh_mean)[0,1]))
# small model: only top outcome features
top = ["spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","te2_mean","te2_ewm","te2_med","te_hh_shrunk","spend_4w_recent","spend_8w","te2_slope","te2_std","te2_zero"]
Xt = df[top].values.astype(float); Xt=np.nan_to_num(Xt,nan=0.0)
mu2,sd2 = Xt[tr].mean(0), Xt[tr].std(0)+1e-9
Xt2=(Xt-mu2)/sd2
for lam in [1,10,100,300,1000]:
    A=Xt2[tr]; I=np.eye(A.shape[1]); I[0,0]=0
    w=np.linalg.solve(A.T@A+lam*I, A.T@y[tr]); pr=Xt2@w
    print("small lam",lam,"val MAE %.3f"%np.abs(pr[va]-y[va]).mean())
# single features
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent"]:
    v = df[c].fillna(df[c].median()).values
    print("single",c,"val MAE %.2f"%np.abs(v[va]-y[va]).mean())


# ---- cell ----
import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
df = saved.merge(tt, on=["household_key","snapshot_day"])
df = df[df.future_spend_4w.notna()]
num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number]).columns.tolist()
X = np.nan_to_num(df[num].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
y = df.future_spend_4w.values
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs = (X-mu)/sd
def ridge(lam):
    A = Xs[tr]; I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@y[tr])
    return Xs@w
best=None
for lam in [1,10,100,300,1000,3000]:
    p = ridge(lam); mae = np.abs(p[va]-y[va]).mean()
    if best is None or mae<best[1]: best=(lam,mae)
print("proxy full E013: lam=%d val431 MAE %.3f" % best)
p = ridge(best[0]); res = y-p
print("resid mean %.2f std %.2f" % (res.mean(), res.std()))
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent","te2_slope"]:
    print("corr(resid,%s)=%.3f" % (c, np.corrcoef(res, df[c].fillna(0))[0,1]))

# single-feature baselines
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent"]:
    v = df[c].fillna(df[c].median()).values
    print("single %-16s val431 MAE %.2f" % (c, np.abs(v[va]-y[va]).mean()))
print("mean-only MAE %.2f" % np.abs(np.full(va.sum(), y[tr].mean())-y[va]).mean())


# ---- cell ----
import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
df = saved.merge(tt, on=["household_key","snapshot_day"])
df = df[df.future_spend_4w.notna()]
num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number]).columns.tolist()
X = np.nan_to_num(df[num].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
y = df.future_spend_4w.values
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs = np.hstack([ (X-mu)/sd, np.ones((len(X),1)) ])
def ridge(lam):
    A = Xs[tr]; I = np.eye(A.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@y[tr])
    return Xs@w
best=None
for lam in [1,10,100,300,1000,3000,10000]:
    p = ridge(lam); mae = np.abs(p[va]-y[va]).mean()
    if best is None or mae<best[1]: best=(lam,mae)
print("proxy full E013: lam=%d val431 MAE %.3f" % best)
p = ridge(best[0]); res = y-p
print("resid mean %.2f std %.2f" % (res.mean(), res.std()))
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent","te2_slope","te2_std","te2_cv","te2_zero","te2_max","te2_min","te2_med","te2_n"]:
    print("corr(resid,%s)=%.3f" % (c, np.corrcoef(res, df[c].fillna(0))[0,1]))


# ---- cell ----
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
    st_trend = ((last4+1)/(prev4+1)).reindex(store_ids).values
    st_tot84 = tx[tx.day>=s-83].groupby("store_id").sales_value.sum().reindex(store_ids).fillna(0).values
    out["st_trend_8w"] = st_trend
    out["st_hh_share"] = hh_main_spend/np.maximum(st_tot84,1.0)
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


# ---- cell ----
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


# ---- cell ----
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
    def wh(h, lo, hi):
        lo=max(lo,1); hi=min(hi,s)
        if hi<lo: return 0.0
        return cum[pos[h],hi]-cum[pos[h],lo-1]
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
    out["st_main_spend_ratio4"] = (wsum_hh_vec(s-27,s)+1)/(wsum_hh_vec(s-55,s-28)+1)
    # stock-up
    bk = tx[tx.day>=s-83].groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","first")).reset_index()
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
    # campaign response
    cc = camps[camps.start_day<=s][["campaign","description","start_day","end_day"]]
    ev = tgts.merge(cc, on="campaign", how="inner")
    ev = ev[(ev.start_day <= s-28) & ev.household_key.isin(idx)]
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

def wsum_hh_vec(lo, hi):
    return None
camps_all = snap.campaigns; tgts_all = snap.campaign_targets; reds_all = snap.coupon_redemptions
feats = {}
for s in SNAPS:
    hhs = saved[saved.snapshot_day==s].household_key.values
    feats[s] = compute_batch1(tx_all, hhs, s, camps_all, tgts_all, reds_all)
b1 = pd.concat([f.assign(snapshot_day=s) for s,f in feats.items()]).reset_index()
print(b1.shape)
api.save_table(b1, "cand_batch1")
print(b1.describe().T[["mean","std","count"]].round(2))


# ---- cell ----
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
    out["st_main_spend_ratio4"] = (wv(s-27,s)+1)/(wv(s-55,s-28)+1)
    bk = tx[tx.day>=s-83].groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","first")).reset_index()
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
    cc = camps[camps.start_day<=s][["campaign","description","start_day","end_day"]]
    ev = tgts.merge(cc, on="campaign", how="inner")
    ev = ev[(ev.start_day <= s-28) & ev.household_key.isin(idx)]
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


# ---- cell ----
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
    out["st_main_spend_ratio4"] = (wv(s-27,s)+1)/(wv(s-55,s-28)+1)
    bk = tx[tx.day>=s-83].groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","first")).reset_index()
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


# ---- cell ----
import agent_api as api, pandas as pd
snap = api.snapshot()
tx = snap.transactions
print("len:", len(tx), "maxday:", tx.day.max() if len(tx) else None)
tx2 = snap.table("transactions")
print("len2:", len(tx2))
h95 = api.load_saved("e013_storeprod.parquet")
h95 = h95[h95.snapshot_day==95].household_key.values[:5]
print(h95)
h = api.history(h95[0])
print("hist rows:", len(h), "days:", h.day.min() if len(h) else None, h.day.max() if len(h) else None)


# ---- cell ----
import agent_api as api, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
saved = api.load_saved("e013_storeprod.parquet")
snap = api.snapshot(); tx_all = snap.transactions
for s in sorted(saved.snapshot_day.unique()):
    tx = tx_all[tx_all.day <= s]
    w84 = tx[tx.day >= s-83]
    hs = w84.groupby(["household_key","store_id"]).sales_value.sum().unstack(fill_value=0.0).reindex(index=saved[saved.snapshot_day==s].household_key.values, fill_value=0.0)
    print(s, "tx", len(tx), "w84", len(w84), "hs.shape", hs.shape, "argmax ok" if hs.shape[1]>0 else "EMPTY COLS")
    if hs.shape[1]==0:
        print("  w84 days:", w84.day.min() if len(w84) else None, w84.day.max() if len(w84) else None)
        print("  tx day range:", tx.day.min(), tx.day.max())


# ---- cell ----
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
