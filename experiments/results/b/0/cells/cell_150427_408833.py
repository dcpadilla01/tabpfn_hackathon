import warnings; warnings.filterwarnings('ignore')

def fn(view, snapshot_day):
    e7 = load_saved('e007_lagseq.parquet')
    sub = e7.loc[e7.snapshot_day == snapshot_day].set_index('household_key').drop(columns=['snapshot_day'])
    tx = view.transactions
    w0 = snapshot_day - 27; w112 = snapshot_day - 111
    last28 = tx[tx.day >= w0]
    last112 = tx[tx.day >= w112]
    spend28_hh = last28.groupby('household_key').sales_value.sum()
    spend112_hh = last112.groupby('household_key').sales_value.sum()
    out = pd.DataFrame(index=sub.index)
    # display / mailer exposure on recently purchased items
    wk0 = (w0 + 8) // 7; wk1 = (snapshot_day + 8) // 7
    dm = view.display_mailer
    dms = dm.loc[(dm.week_no >= wk0) & (dm.week_no <= wk1), ['product_id','store_id','week_no','display','mailer']]
    j = last28.merge(dms, on=['product_id','store_id','week_no'], how='left')
    disp = j.display.notna() & (j.display != 0)
    mail = j.mailer.notna() & (j.mailer != '0')
    out['disp_spend28'] = j.loc[disp].groupby('household_key').sales_value.sum()
    out['mail_spend28'] = j.loc[mail].groupby('household_key').sales_value.sum()
    out['disp_share28'] = out['disp_spend28'] / spend28_hh
    out['mail_share28'] = out['mail_spend28'] / spend28_hh
    out['disp_rows28'] = j.loc[disp].groupby('household_key').size()
    out['mail_rows28'] = j.loc[mail].groupby('household_key').size()
    out['nlines28'] = j.groupby('household_key').size()
    # private-brand share of last-28d spend
    br = view.products[['product_id','brand']]
    jb = last28.merge(br, on='product_id', how='left')
    priv = jb.loc[jb.brand == 'Private'].groupby('household_key').sales_value.sum()
    out['brand_priv_spend28'] = priv
    out['brand_priv_share28'] = priv / spend28_hh
    # repeat-commodity spend (commodity bought before within 112d window)
    cm = view.products[['product_id','commodity_desc']]
    j3 = last112.merge(cm, on='product_id', how='left')
    first_d = j3.groupby(['household_key','commodity_desc']).day.transform('min')
    ret = j3[j3.day > first_d]
    out['dsp_rets112'] = ret.groupby('household_key').sales_value.sum()
    out['dsp_rets28'] = ret.loc[ret.day >= w0].groupby('household_key').sales_value.sum()
    out['dsp_rets_share112'] = out['dsp_rets112'] / spend112_hh
    return out

fe = build_features(fn)
print(fe.shape)
print(fe.head(3).T.head(30))
print(fe.groupby('snapshot_day').size())
p = save_table(fe, 'e008_exposure')
print(p)