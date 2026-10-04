import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    hh = view.households
    tx = view.transactions
    prods = view.products
    # ----- marketing exposure -----
    ct = view.campaign_targets
    camps = view.campaigns
    red = view.coupon_redemptions

    targeted = ct.groupby('household_key').size().rename('n_campaigns_targeted')
    tA = ct[ct.description=='TypeA'].groupby('household_key').size().rename('tA')
    tB = ct[ct.description=='TypeB'].groupby('household_key').size().rename('tB')
    tC = ct[ct.description=='TypeC'].groupby('household_key').size().rename('tC')
    # active campaign right now?
    active = camps[(camps.start_day<=snapshot_day)&(camps.end_day>=snapshot_day)]
    act_hh = ct[ct.campaign.isin(active.campaign)].household_key.unique()
    mkt = pd.concat([targeted,tA,tB,tC],axis=1)
    mkt['targeted'] = 1
    mkt['active_campaign'] = 0
    mkt.loc[mkt.index.isin(act_hh),'active_campaign'] = 1
    # days since most recent campaign start that targeted this hh
    starts = ct.merge(camps[['campaign','start_day']],on='campaign')
    last_start = starts.groupby('household_key').start_day.max()
    mkt['days_since_camp_start'] = (snapshot_day - last_start).clip(lower=0)

    # coupon redemption history
    r_tot = red.groupby('household_key').size().rename('n_redeem_life')
    r_84 = red[red.day>snapshot_day-84].groupby('household_key').size().rename('n_redeem_84')
    last_red = red.groupby('household_key').day.max()
    mkt = mkt.join(r_tot).join(r_84)
    mkt['n_redeem_life'] = mkt.n_redeem_life.fillna(0)
    mkt['n_redeem_84'] = mkt.n_redeem_84.fillna(0)
    dr = (snapshot_day - last_red).reindex(mkt.index)
    mkt['days_since_redeem'] = dr.fillna(9999).clip(upper=9999)

    # ----- discount usage & mix from transactions -----
    tx = tx.merge(prods[['product_id','department','brand']],on='product_id',how='left')
    tx['t28'] = tx.day > snapshot_day-28
    tx['t84'] = tx.day > snapshot_day-84
    tx['t364'] = tx.day > snapshot_day-364
    g = tx.groupby('household_key')
    disc28 = g.apply(lambda d: d.loc[d.t28,['coupon_disc','retail_disc','coupon_match_disc']].sum().sum())
    spend28 = g.apply(lambda d: d.loc[d.t28,'sales_value'].sum())
    mkt['disc_share_28'] = (-disc28/(spend28+1)).clip(upper=1)
    # private brand share of spend 84d
    priv84 = g.apply(lambda d: d.loc[d.t84 & (d.brand=='Private'),'sales_value'].sum())
    spend84 = g.apply(lambda d: d.loc[d.t84,'sales_value'].sum())
    mkt['private_share_84'] = (priv84/(spend84+1)).clip(upper=1)
    # dept shares of spend 84d (top departments)
    top = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','DELI','PASTRY']
    for dep in top:
        s = g.apply(lambda d, dep=dep: d.loc[d.t84 & (d.department==dep),'sales_value'].sum())
        mkt['sh_'+dep[:6].lower()] = (s/(spend84+1)).clip(upper=1)
    mkt = mkt.reset_index()
    mkt.columns.name=None
    return mkt.set_index('household_key')

out = agent_api.build_features(fn)
print(out.shape)
print(out.columns.tolist())
print(out.isna().mean().round(3).to_string())
path = agent_api.save_table(out,'e002_marketing')
print(path)
