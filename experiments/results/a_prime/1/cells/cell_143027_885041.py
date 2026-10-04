import agent_api as api
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.transactions
    tx = tx[tx.day <= d]
    firstd = tx.groupby('household_key').day.min()
    hh = pd.Index(firstd[firstd <= d-84].index, name='household_key')
    eps = 1e-9
    f = pd.DataFrame(index=hh)
    for k in range(1, 7):
        lo = d - 28*k + 1; hi = d - 28*(k-1)
        w = tx[(tx.day >= lo) & (tx.day <= hi)]
        a = w.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'),
                                           da=('day','nunique'), q=('quantity','sum'),
                                           st=('store_id','nunique'), pr=('product_id','nunique'))
        a = a.reindex(hh)
        f[f'spend_l{k}'] = a.sp.fillna(0.0)
        if k <= 3:
            f[f'trips_l{k}'] = a.tr.fillna(0.0)
            f[f'days_active_l{k}'] = a.da.fillna(0.0)
            f[f'avg_basket_l{k}'] = (a.sp/a.tr).fillna(0.0)
            f[f'stores_l{k}'] = a.st.fillna(0.0)
            f[f'prods_l{k}'] = a.pr.fillna(0.0)
    lo = d - 28*13 + 1; hi = d - 28*12
    w = tx[(tx.day >= lo) & (tx.day <= hi)] if lo > 0 else tx.iloc[:0]
    f['spend_l13'] = w.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    a = tx.groupby('household_key').agg(firstd=('day','min'), lastd=('day','max'), sp=('sales_value','sum')).reindex(hh)
    f['tenure'] = d - a.firstd + 1
    f['days_since_last'] = d - a.lastd
    f['spend_total'] = a.sp.fillna(0.0)
    f['spend_rate28'] = a.sp.fillna(0.0)/f['tenure']*28
    w = tx[tx.day > d-84]
    a = w.groupby('household_key').agg(cd=('coupon_disc', lambda s: -s[s<0].sum()),
                                       rd=('retail_disc', lambda s: -s[s<0].sum()),
                                       md=('coupon_match_disc', lambda s: -s[s<0].sum()),
                                       tt=('trans_time','mean')).reindex(hh)
    sp84 = f['spend_l3']
    f['disc84'] = (a.cd.fillna(0)+a.rd.fillna(0)+a.md.fillna(0))
    f['disc_share84'] = f['disc84']/(sp84+f['disc84']+eps)
    f['evening_share84'] = w.assign(ev=(w.trans_time>=1700).astype(float)).groupby('household_key').ev.mean().reindex(hh)
    f['trend_1v2'] = f['spend_l1']/(f['spend_l2']+eps)
    f['trend_1v3'] = f['spend_l1']/(f['spend_l3']+eps)
    f['spend_l123_mean'] = (f['spend_l1']+f['spend_l2']+f['spend_l3'])/3
    f['spend_l456_mean'] = (f['spend_l4']+f['spend_l5']+f['spend_l6'])/3
    f['momentum'] = f['spend_l123_mean'] - f['spend_l456_mean']
    f['active_share_l1'] = f['days_active_l1']/28
    f['zero_recent'] = (f['spend_l1']<=0).astype(float)
    return f

full = api.build_features(fn)
print(full.shape, full.isna().mean().mean().round(4))
tg = api.train_targets()
m = full.merge(tg, on=['household_key','snapshot_day'])
print(m[['spend_l1','spend_l2','spend_l3','spend_l123_mean','spend_rate28','spend_l13','momentum','future_spend_4w']].corr()['future_spend_4w'].round(3))
path = api.save_table(full, 'e001_txhist')
print(path)
