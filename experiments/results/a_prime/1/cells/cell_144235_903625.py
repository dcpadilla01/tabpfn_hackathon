import agent_api as api
import pandas as pd, numpy as np

def feat(view, day):
    hh = view.households
    if isinstance(hh, pd.DataFrame):
        hh_ids = pd.Index(hh['household_key'].values) if 'household_key' in hh.columns else pd.Index(hh.index)
    else:
        hh_ids = pd.Index(np.asarray(hh).ravel())
    out = pd.DataFrame(index=hh_ids)

    camps_all = view.campaigns
    camps = camps_all[camps_all.start_day <= day]
    desc = camps.set_index('campaign')['description']
    start = camps.set_index('campaign')['start_day']

    ct = view.campaign_targets
    ct = ct[ct.campaign.isin(camps.campaign)].copy()
    ct['desc'] = ct.campaign.map(desc)
    ct['start_day'] = ct.campaign.map(start)

    g = ct.groupby('household_key')
    out['mkt_n_campaigns_targeted'] = g.size().reindex(hh_ids).fillna(0)
    out['mkt_ever_targeted'] = (out['mkt_n_campaigns_targeted'] > 0).astype(int)
    for d in ['TypeA','TypeB','TypeC']:
        s = ct[ct.desc==d].groupby('household_key').size()
        out['mkt_tgt_'+d+'_n'] = s.reindex(hh_ids).fillna(0)
    last_start = g['start_day'].max()
    first_start = g['start_day'].min()
    ds_last = (day - last_start).reindex(hh_ids)
    ds_first = (day - first_start).reindex(hh_ids)
    out['mkt_days_since_last_target'] = ds_last.clip(upper=365).fillna(365)
    out['mkt_days_since_first_target'] = ds_first.clip(upper=365).fillna(365)
    out['mkt_targeted_last28'] = (ds_last <= 28).fillna(False).astype(int)
    out['mkt_targeted_last84'] = (ds_last <= 84).fillna(False).astype(int)

    active = camps[(camps.start_day <= day) & (camps.end_day >= day)]
    act_ct = ct[ct.campaign.isin(active.campaign)]
    out['mkt_n_active_campaigns'] = act_ct.groupby('household_key').size().reindex(hh_ids).fillna(0)

    fut = camps_all[camps_all.end_day >= day+1]
    fut_ct = ct[ct.campaign.isin(fut.campaign)]
    out['mkt_n_future_campaigns'] = fut_ct.groupby('household_key').size().reindex(hh_ids).fillna(0)

    cr = view.coupon_redemptions.copy()
    cr['desc'] = cr.campaign.map(desc)
    gcr = cr.groupby('household_key')
    out['mkt_n_redemptions'] = gcr.size().reindex(hh_ids).fillna(0)
    out['mkt_redemptions_28'] = cr[cr.day > day-28].groupby('household_key').size().reindex(hh_ids).fillna(0)
    out['mkt_redemptions_84'] = cr[cr.day > day-84].groupby('household_key').size().reindex(hh_ids).fillna(0)
    ds_red = (day - gcr['day'].max()).reindex(hh_ids)
    out['mkt_days_since_last_redemption'] = ds_red.clip(upper=365).fillna(365)
    for d in ['TypeA','TypeB','TypeC']:
        s = cr[cr.desc==d].groupby('household_key').size()
        out['mkt_red_'+d] = s.reindex(hh_ids).fillna(0)
    out['mkt_red_per_campaign'] = out['mkt_n_redemptions'] / (1.0 + out['mkt_n_campaigns_targeted'])
    return out

df = api.build_features(feat)
print(df.shape)
print(df.describe().T[['mean','std','min','max']].round(2))
path = api.save_table(df, 'e004_mkt.parquet')
print(path)