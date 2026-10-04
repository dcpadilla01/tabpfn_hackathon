import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def mk(view, day):
    hh = view.households
    idx = pd.Index(hh, name='household_key')
    f = pd.DataFrame(index=idx)
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(hh))]
    g = tx.groupby('household_key')
    f['tenure'] = (day - g.day.min()).reindex(idx)
    t28 = tx[tx.day > day-28]
    g28 = t28.groupby('household_key')
    sp28 = g28.sales_value.sum().reindex(idx, fill_value=0)
    q28 = g28.quantity.sum().reindex(idx, fill_value=0)
    d28 = (g28.coupon_disc.sum()+g28.retail_disc.sum()+g28.coupon_match_disc.sum()).reindex(idx, fill_value=0)
    f['disc_share_28'] = np.where(sp28>0, d28/sp28.replace(0,np.nan), 0)
    f['unit_price_28'] = np.where(q28>0, sp28/q28.replace(0,np.nan), np.nan)
    f['stores_28'] = g28.store_id.nunique().reindex(idx, fill_value=0)
    f['weekend_share_28'] = g28.day.apply(lambda s: np.mean([(d%7)>=5 for d in s])).reindex(idx, fill_value=0)
    t84 = tx[tx.day > day-84]
    g84 = t84.groupby('household_key')
    f['basket_mean_84'] = (g84.sales_value.sum()/g84.basket_id.nunique()).reindex(idx)
    f['trips_84'] = g84.basket_id.nunique().reindex(idx, fill_value=0)
    f['active_weeks_28'] = g28.week_no.nunique().reindex(idx, fill_value=0)
    f['tenure_x_freq'] = f['tenure']/ (f['trips_84'].replace(0,np.nan))
    f['spend_per_active_day_28'] = sp28 / f['active_weeks_28'].replace(0,np.nan)
    return f

C = A.build_features(mk)
tt = A.train_targets()
tr = C.merge(tt, on=['household_key','snapshot_day'])
base = A.load_saved("temporal_structure.parquet")
feat = [c for c in base.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(base[c])]
B = tr[['household_key','snapshot_day']].merge(base[['household_key','snapshot_day']+feat], on=['household_key','snapshot_day'])
Xb = B[feat].replace([np.inf,-np.inf], np.nan)
Xb = Xb.fillna(Xb.median())
y = tr.future_spend_4w.values
hh_arr = tr.household_key.values
u = pd.unique(hh_arr); rng = np.random.RandomState(0); rng.shuffle(u)
folds = {h: i%5 for i,h in enumerate(u)}
oof = np.zeros(len(y))
Xm = Xb.values; mu, sd = Xm.mean(0), Xm.std(0)+1e-9
Xz = (Xm-mu)/sd
for k in range(5):
    va = np.array([folds[h]==k for h in hh_arr]); trm = ~va
    Z = np.hstack([Xz[trm], np.ones((trm.sum(),1))])
    beta = np.linalg.solve(Z.T@Z + 10*np.eye(Z.shape[1]), Z.T@y[trm])
    Zv = np.hstack([Xz[va], np.ones((va.sum(),1))])
    oof[va] = Zv@beta
res = y - oof
print("ridge OOF MAE proxy:", np.mean(np.abs(res)))
cand = [c for c in C.columns if c not in ('household_key','snapshot_day')]
Cc = tr[['household_key','snapshot_day']].merge(C, on=['household_key','snapshot_day'])
for c in cand:
    x = pd.to_numeric(Cc[c], errors='coerce').replace([np.inf,-np.inf],np.nan)
    x = x.fillna(x.median()).values
    print(f"{c:26s} corr_res={np.corrcoef(x,res)[0,1]:+.4f} corr_y={np.corrcoef(x,y)[0,1]:+.4f}")
path = A.save_table(C, "behavioral_candidates.parquet")
print(path)
