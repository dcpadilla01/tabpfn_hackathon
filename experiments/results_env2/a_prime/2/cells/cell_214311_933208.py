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
