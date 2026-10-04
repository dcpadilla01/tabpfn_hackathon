import agent_api as A
v = A.snapshot()
print("== campaigns", v.campaigns.shape); print(v.campaigns.head(3).to_string())
print("== campaign_targets", v.campaign_targets.shape); print(v.campaign_targets.head(3).to_string())
print("== coupon_redemptions", v.coupon_redemptions.shape); print(v.coupon_redemptions.head(3).to_string())
print("== display_mailer", v.display_mailer.shape); print(v.display_mailer.head(3).to_string())
print(v.campaign_targets.description.value_counts())
print(v.display_mailer.display.value_counts().head(10))
print(v.display_mailer.mailer.value_counts().head(10))
dm = v.display_mailer
print("dm stores:", dm.store_id.nunique(), "weeks:", dm.week_no.nunique())
red = v.coupon_redemptions
print("redemptions households:", red.household_key.nunique(), "rows:", len(red))
ct = v.campaign_targets
print("targeted households:", ct.household_key.nunique(), "rows:", len(ct))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
v = A.snapshot()
print(v.campaigns.to_string())
red = v.coupon_redemptions
print("redemption days:", red.day.min(), red.day.max())
print(red.groupby('campaign').size())
# how many households needing rows have ever been targeted / redeemed?
hh = v.households
print("households needing rows:", len(hh))
ct = v.campaign_targets
print("targeted overlap:", hh.isin(ct.household_key).sum())
print("redeem overlap:", hh.isin(red.household_key).sum())
# transactions size
print("tx rows up to 459:", len(v.transactions))
tx = v.transactions
print("tx per household median:", tx.groupby('household_key').size().median())
# quick signal check: campaigns targeted in last 56 days vs train target
tt = A.train_targets()
print(tt.shape, tt.head())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
for p in ["temporal_structure.parquet","recency_agg.parquet","dept_mix_recency.parquet"]:
    df = A.load_saved(p)
    print(p, df.shape)
    print(list(df.columns)[:40])
tt = A.train_targets()
print(tt.shape); print(tt.head(3)); print(tt.future_spend_4w.describe())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def mk_features(view, day):
    hh = view.households
    idx = pd.Index(hh, name='household_key')
    f = pd.DataFrame(index=idx)
    camps = view.campaigns
    active = camps[(camps.start_day <= day) & (camps.end_day >= day)]
    recent = camps[(camps.start_day <= day) & (camps.start_day > day-56)]
    ct = view.campaign_targets
    ca = ct[ct.campaign.isin(set(active.campaign))]
    cr = ct[ct.campaign.isin(set(recent.campaign))]
    f['n_camp_active'] = ca.groupby('household_key').size().reindex(idx, fill_value=0)
    f['n_camp_recent'] = cr.groupby('household_key').size().reindex(idx, fill_value=0)
    for t in ['TypeA','TypeB','TypeC']:
        f['camp_'+t+'_recent'] = cr[cr.description==t].groupby('household_key').size().reindex(idx, fill_value=0)
    red = view.coupon_redemptions
    red_r = red[red.day > day-56]
    f['red_cnt_56'] = red_r.groupby('household_key').size().reindex(idx, fill_value=0)
    last_red = red.groupby('household_key').day.max()
    f['days_since_red'] = (day - last_red).reindex(idx)
    f['ever_red'] = f['days_since_red'].notna().astype(float)
    # display/mailer exposure of recent purchases
    tx = view.transactions
    tx28 = tx[tx.day > day-28][['household_key','product_id','store_id','day','sales_value']].copy()
    if len(tx28):
        tx28['week_no'] = (tx28.day + 8)//7
        dm = view.display_mailer
        m = tx28.merge(dm, on=['product_id','store_id','week_no'], how='left')
        m['disp'] = (m.display.astype(str) != '0').astype(float)
        m['mail'] = (m.mailer.astype(str) != '0').astype(float)
        g = m.groupby('household_key')
        spend = tx28.groupby('household_key').sales_value.sum().reindex(idx, fill_value=0)
        f['disp_spend_28'] = g.apply(lambda x: (x.sales_value*x.disp).sum()).reindex(idx, fill_value=0)
        f['mail_spend_28'] = g.apply(lambda x: (x.sales_value*x.mail).sum()).reindex(idx, fill_value=0)
        f['disp_share_28'] = np.where(spend>0, f['disp_spend_28']/spend.replace(0,np.nan), 0)
        f['mail_share_28'] = np.where(spend>0, f['mail_spend_28']/spend.replace(0,np.nan), 0)
        f['disp_lines_28'] = g.disp.sum().reindex(idx, fill_value=0)
    return f

X = A.build_features(mk_features)
print(X.shape)
tt = A.train_targets()
tr = X.merge(tt, on=['household_key','snapshot_day'])
print(tr.shape)
num = X.columns.drop(['household_key','snapshot_day'])
for c in num:
    r = np.corrcoef(tr[c].fillna(tr[c].median()).astype(float), tr.future_spend_4w)[0,1]
    print(f"{c:20s} corr={r:+.3f} mean={tr[c].mean():.2f} nonzero={np.mean(tr[c]!=0):.2f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def partial_corr(x, y, z):
    # corr of residuals of x~z and y~z (z = list of controls)
    Z = np.column_stack([np.ones(len(z))] + [z[c].values for c in z.columns])
    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values.astype(float)), res(y.values.astype(float))
    return np.corrcoef(rx, ry)[0,1]

tt = A.train_targets()
tr = X.merge(tt, on=['household_key','snapshot_day'])
base = A.load_saved("temporal_structure.parquet")
trb = tr.merge(base.drop(columns=[c for c in base.columns if c.startswith(('spend_w','trips_w','lines_w','qty_w','disc_w')) or c in ('basket_mean_w56','basket_max_w56','stores_w84','spend_prev28','spend_ratio_28_56')], errors='ignore'), on=['household_key','snapshot_day'], suffixes=('','_b'))
print(trb.shape)
y = tr.future_spend_4w
ctrl = trb[['spend_28','spend_56','spend_84','trips_28','days_since_last']].copy()
for c in ['disp_spend_28','mail_spend_28','disp_share_28','mail_share_28','disp_lines_28','n_camp_active','n_camp_recent','camp_TypeA_recent','ever_red','red_cnt_56']:
    print(f"{c:18s} partial={partial_corr(tr[c].fillna(0), y, ctrl):+.4f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z))] + [z[c].values.astype(float) for c in z.columns])
    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values.astype(float)), res(y.values.astype(float))
    return np.corrcoef(rx, ry)[0,1]

tt = A.train_targets()
X = A.build_features(lambda v, d: pd.DataFrame(index=pd.Index(v.households, name='household_key'), data={'z':1.0}))  # placeholder not needed; rebuild marketing features inline


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def mk_features(view, day):
    hh = view.households
    idx = pd.Index(hh, name='household_key')
    f = pd.DataFrame(index=idx)
    camps = view.campaigns
    active = camps[(camps.start_day <= day) & (camps.end_day >= day)]
    recent = camps[(camps.start_day <= day) & (camps.start_day > day-56)]
    ct = view.campaign_targets
    ca = ct[ct.campaign.isin(set(active.campaign))]
    cr = ct[ct.campaign.isin(set(recent.campaign))]
    f['n_camp_active'] = ca.groupby('household_key').size().reindex(idx, fill_value=0)
    f['n_camp_recent'] = cr.groupby('household_key').size().reindex(idx, fill_value=0)
    for t in ['TypeA','TypeB','TypeC']:
        f['camp_'+t+'_recent'] = cr[cr.description==t].groupby('household_key').size().reindex(idx, fill_value=0)
    red = view.coupon_redemptions
    red_r = red[red.day > day-56]
    f['red_cnt_56'] = red_r.groupby('household_key').size().reindex(idx, fill_value=0)
    last_red = red.groupby('household_key').day.max().reindex(idx)
    f['days_since_red'] = (day - last_red)
    f['ever_red'] = f['days_since_red'].notna().astype(float)
    tx = view.transactions
    tx28 = tx[tx.day > day-28][['household_key','product_id','store_id','day','sales_value']].copy()
    tx28['week_no'] = (tx28.day + 8)//7
    dm = view.display_mailer
    m = tx28.merge(dm, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m.display.astype(str) != '0').astype(float)
    m['mail'] = (m.mailer.astype(str) != '0').astype(float)
    g = m.groupby('household_key')
    spend = tx28.groupby('household_key').sales_value.sum().reindex(idx, fill_value=0)
    f['disp_spend_28'] = g.apply(lambda x: (x.sales_value*x.disp).sum()).reindex(idx, fill_value=0)
    f['mail_spend_28'] = g.apply(lambda x: (x.sales_value*x.mail).sum()).reindex(idx, fill_value=0)
    f['disp_share_28'] = np.where(spend>0, f['disp_spend_28']/spend.replace(0,np.nan), 0)
    f['mail_share_28'] = np.where(spend>0, f['mail_spend_28']/spend.replace(0,np.nan), 0)
    f['disp_lines_28'] = g.disp.sum().reindex(idx, fill_value=0)
    return f

X = A.build_features(mk_features)
tt = A.train_targets()
tr = X.merge(tt, on=['household_key','snapshot_day'])

def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z))] + [z[c].values.astype(float) for c in z.columns])
    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values.astype(float)), res(y.values.astype(float))
    return np.corrcoef(rx, ry)[0,1]

base = A.load_saved("temporal_structure.parquet")
core = ['spend_28','spend_56','spend_84','spend_112','trips_28','trips_56','days_since_last','spend_ratio_28_56','avg_basket_28']
ctrl = base[core]
y = tr.future_spend_4w
for c in ['disp_spend_28','mail_spend_28','disp_share_28','mail_share_28','disp_lines_28','n_camp_active','n_camp_recent','camp_TypeA_recent','ever_red','red_cnt_56']:
    print(f"{c:18s} partial={partial_corr(tr[c].fillna(0), y, ctrl):+.4f}")
path = A.save_table(X, "marketing_exposure.parquet")
print(path)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def mk_features(view, day):
    hh = view.households
    idx = pd.Index(hh, name='household_key')
    f = pd.DataFrame(index=idx)
    camps = view.campaigns
    active = camps[(camps.start_day <= day) & (camps.end_day >= day)]
    recent = camps[(camps.start_day <= day) & (camps.start_day > day-56)]
    ct = view.campaign_targets
    ca = ct[ct.campaign.isin(set(active.campaign))]
    cr = ct[ct.campaign.isin(set(recent.campaign))]
    f['n_camp_active'] = ca.groupby('household_key').size().reindex(idx, fill_value=0)
    f['n_camp_recent'] = cr.groupby('household_key').size().reindex(idx, fill_value=0)
    for t in ['TypeA','TypeB','TypeC']:
        f['camp_'+t+'_recent'] = cr[cr.description==t].groupby('household_key').size().reindex(idx, fill_value=0)
    red = view.coupon_redemptions
    red_r = red[red.day > day-56]
    f['red_cnt_56'] = red_r.groupby('household_key').size().reindex(idx, fill_value=0)
    last_red = red.groupby('household_key').day.max().reindex(idx)
    f['days_since_red'] = (day - last_red)
    f['ever_red'] = f['days_since_red'].notna().astype(float)
    tx = view.transactions
    tx28 = tx[tx.day > day-28][['household_key','product_id','store_id','day','sales_value']].copy()
    tx28['week_no'] = (tx28.day + 8)//7
    dm = view.display_mailer
    m = tx28.merge(dm, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m.display.astype(str) != '0').astype(float)
    m['mail'] = (m.mailer.astype(str) != '0').astype(float)
    g = m.groupby('household_key')
    spend = tx28.groupby('household_key').sales_value.sum().reindex(idx, fill_value=0)
    f['disp_spend_28'] = g.apply(lambda x: (x.sales_value*x.disp).sum()).reindex(idx, fill_value=0)
    f['mail_spend_28'] = g.apply(lambda x: (x.sales_value*x.mail).sum()).reindex(idx, fill_value=0)
    f['disp_share_28'] = np.where(spend>0, f['disp_spend_28']/spend.replace(0,np.nan), 0)
    f['mail_share_28'] = np.where(spend>0, f['mail_spend_28']/spend.replace(0,np.nan), 0)
    f['disp_lines_28'] = g.disp.sum().reindex(idx, fill_value=0)
    return f

X = A.build_features(mk_features)
tt = A.train_targets()
tr = X.merge(tt, on=['household_key','snapshot_day'])

def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z))] + [z[c].values.astype(float) for c in z.columns])
    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values.astype(float)), res(y.values.astype(float))
    return np.corrcoef(rx, ry)[0,1]

base = A.load_saved("temporal_structure.parquet")
core = ['spend_28','spend_56','spend_84','spend_112','trips_28','trips_56','days_since_last','spend_ratio_28_56','avg_basket_28']
ctrl = base.merge(tt, on=['household_key','snapshot_day'])[['household_key','snapshot_day']+core]
ctrl = tr[['household_key','snapshot_day']].merge(ctrl, on=['household_key','snapshot_day'])[core]
y = tr.future_spend_4w
for c in ['disp_spend_28','mail_spend_28','disp_share_28','mail_share_28','disp_lines_28','n_camp_active','n_camp_recent','camp_TypeA_recent','ever_red','red_cnt_56']:
    print(f"{c:18s} partial={partial_corr(tr[c].fillna(0), y, ctrl):+.4f}")
path = A.save_table(X, "marketing_exposure.parquet")
print(path)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def mk_features(view, day):
    hh = view.households
    idx = pd.Index(hh, name='household_key')
    f = pd.DataFrame(index=idx)
    camps = view.campaigns
    active = camps[(camps.start_day <= day) & (camps.end_day >= day)]
    recent = camps[(camps.start_day <= day) & (camps.start_day > day-56)]
    ct = view.campaign_targets
    ca = ct[ct.campaign.isin(set(active.campaign))]
    cr = ct[ct.campaign.isin(set(recent.campaign))]
    f['n_camp_active'] = ca.groupby('household_key').size().reindex(idx, fill_value=0)
    f['n_camp_recent'] = cr.groupby('household_key').size().reindex(idx, fill_value=0)
    for t in ['TypeA','TypeB','TypeC']:
        f['camp_'+t+'_recent'] = cr[cr.description==t].groupby('household_key').size().reindex(idx, fill_value=0)
    red = view.coupon_redemptions
    red_r = red[red.day > day-56]
    f['red_cnt_56'] = red_r.groupby('household_key').size().reindex(idx, fill_value=0)
    last_red = red.groupby('household_key').day.max().reindex(idx)
    f['days_since_red'] = (day - last_red)
    f['ever_red'] = f['days_since_red'].notna().astype(float)
    tx = view.transactions
    tx28 = tx[tx.day > day-28][['household_key','product_id','store_id','day','sales_value']].copy()
    tx28['week_no'] = (tx28.day + 8)//7
    dm = view.display_mailer
    m = tx28.merge(dm, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m.display.astype(str) != '0').astype(float)
    m['mail'] = (m.mailer.astype(str) != '0').astype(float)
    g = m.groupby('household_key')
    spend = tx28.groupby('household_key').sales_value.sum().reindex(idx, fill_value=0)
    f['disp_spend_28'] = g.apply(lambda x: (x.sales_value*x.disp).sum()).reindex(idx, fill_value=0)
    f['mail_spend_28'] = g.apply(lambda x: (x.sales_value*x.mail).sum()).reindex(idx, fill_value=0)
    f['disp_share_28'] = np.where(spend>0, f['disp_spend_28']/spend.replace(0,np.nan), 0)
    f['mail_share_28'] = np.where(spend>0, f['mail_spend_28']/spend.replace(0,np.nan), 0)
    f['disp_lines_28'] = g.disp.sum().reindex(idx, fill_value=0)
    return f

X = A.build_features(mk_features)
tt = A.train_targets()
tr = X.merge(tt, on=['household_key','snapshot_day'])

def partial_corr(x, y, z):
    z = z.replace([np.inf,-np.inf], np.nan).fillna(z.median())
    Z = np.column_stack([np.ones(len(z))] + [z[c].values.astype(float) for c in z.columns])
    def res(v):
        v = np.nan_to_num(np.asarray(v, float), nan=np.nanmean(np.asarray(v,float)))
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values), res(y.values)
    return np.corrcoef(rx, ry)[0,1]

base = A.load_saved("temporal_structure.parquet")
core = ['spend_28','spend_56','spend_84','spend_112','trips_28','trips_56','days_since_last','spend_ratio_28_56','avg_basket_28']
ctrl = tr[['household_key','snapshot_day']].merge(base[['household_key','snapshot_day']+core], on=['household_key','snapshot_day'])[core]
y = tr.future_spend_4w
for c in ['disp_spend_28','mail_spend_28','disp_share_28','mail_share_28','disp_lines_28','n_camp_active','n_camp_recent','camp_TypeA_recent','ever_red','red_cnt_56']:
    print(f"{c:18s} partial={partial_corr(tr[c].fillna(0), y, ctrl):+.4f}")
path = A.save_table(X, "marketing_exposure.parquet")
print(path)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def mk(view, day):
    hh = view.households
    idx = pd.Index(hh, name='household_key')
    f = pd.DataFrame(index=idx)
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(hh))]
    g = tx.groupby('household_key')
    f['tenure'] = (day - g.day.min()).reindex(idx)          # days since first purchase
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
feat = [c for c in base.columns if c not in ('household_key','snapshot_day')]
B = tr[['household_key','snapshot_day']].merge(base, on=['household_key','snapshot_day'])
Xb = B[feat].apply(lambda s: s.astype(float)).replace([np.inf,-np.inf], np.nan)
Xb = Xb.fillna(Xb.median())
y = tr.future_spend_4w.values
# 5-fold grouped CV ridge to get OOF residuals
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
print("ridge OOF MAE (proxy):", np.mean(np.abs(res)))
cand = [c for c in C.columns if c not in ('household_key','snapshot_day')]
Cc = tr[['household_key','snapshot_day']].merge(C, on=['household_key','snapshot_day'])
for c in cand:
    x = pd.to_numeric(Cc[c], errors='coerce').replace([np.inf,-np.inf],np.nan)
    x = x.fillna(x.median()).values
    print(f"{c:26s} corr_res={np.corrcoef(x,res)[0,1]:+.4f} corr_y={np.corrcoef(x,y)[0,1]:+.4f}")


# ---- cell ----
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


# ---- cell ----
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
Xb = B[feat].replace([np.inf,-np.inf], np.nan).astype(float)
Xb = Xb.fillna(Xb.median()).fillna(0)
y = tr.future_spend_4w.values.astype(float)
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
