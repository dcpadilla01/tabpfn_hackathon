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
