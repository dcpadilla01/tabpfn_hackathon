
import numpy as np, pandas as pd

def make_features(view, snapshot_day):
    tx = view.transactions
    first_day = tx.groupby('household_key')['day'].min()
    hh = pd.Index(first_day[first_day <= snapshot_day - 84].index, name='household_key')
    print("inside fn: households attr =", type(view.households), "| elig:", len(hh))
    t = tx[tx.household_key.isin(hh)]
    feats = pd.DataFrame(index=hh)

    # --- shopping time-of-day habits (182d) ---
    tm = t[['household_key','trans_time','day']].copy()
    tm = tm[tm['day'] > snapshot_day - 182]
    tm['mins'] = (tm['trans_time'] // 100) * 60 + (tm['trans_time'] % 100)
    gm = tm.groupby('household_key')['mins']
    feats['tod_mean_182'] = gm.mean()
    feats['tod_std_182'] = gm.std()
    feats['evening_share_182'] = tm.assign(e=(tm.mins >= 17*60).astype(float)).groupby('household_key')['e'].mean()
    feats['morning_share_182'] = tm.assign(m=(tm.mins < 10*60).astype(float)).groupby('household_key')['m'].mean()

    # --- discount / coupon usage in purchases ---
    for w in (84, 182):
        tw = t[t['day'] > snapshot_day - w]
        gg = tw.groupby('household_key')
        sp = gg['sales_value'].sum().replace(0, np.nan)
        cd = (gg['coupon_disc'].sum() + gg['coupon_match_disc'].sum()).abs()
        rd = gg['retail_disc'].sum().abs()
        feats[f'coupon_share_{w}'] = cd / sp
        feats[f'retail_share_{w}'] = rd / sp

    # --- campaign targeting ---
    ct = view.campaign_targets
    camp = view.campaigns
    if len(ct) and len(camp):
        ct2 = ct.merge(camp[['campaign','start_day','end_day']], on='campaign', how='left')
        for typ in ('TypeA','TypeB','TypeC'):
            sub = ct2[ct2.description == typ]
            feats[f'camp_n_{typ}'] = sub.groupby('household_key').size().reindex(hh).fillna(0.0)
        feats['camp_days_since_start'] = (snapshot_day - ct2.groupby('household_key')['start_day'].max()).reindex(hh)
        act = ct2[(ct2.start_day <= snapshot_day) & (ct2.end_day >= snapshot_day)]
        feats['camp_active_n'] = act.groupby('household_key').size().reindex(hh).fillna(0.0)
    else:
        for typ in ('TypeA','TypeB','TypeC'):
            feats[f'camp_n_{typ}'] = 0.0
        feats['camp_days_since_start'] = np.nan
        feats['camp_active_n'] = 0.0

    # --- coupon redemption engagement ---
    red = view.coupon_redemptions
    red = red[red.household_key.isin(hh)]
    if len(red):
        for w in (28, 84, 182, 365):
            feats[f'red_n_{w}'] = red[red.day > snapshot_day - w].groupby('household_key').size().reindex(hh).fillna(0.0)
        feats['red_days_since'] = (snapshot_day - red.groupby('household_key')['day'].max()).reindex(hh)
    else:
        for w in (28, 84, 182, 365):
            feats[f'red_n_{w}'] = 0.0
        feats['red_days_since'] = np.nan
    feats['red_ever'] = feats['red_days_since'].notna().astype(float)
    tot_camp = feats[['camp_n_TypeA','camp_n_TypeB','camp_n_TypeC']].sum(axis=1)
    feats['red_rate_365'] = feats['red_n_365'] / tot_camp.replace(0, np.nan)
    return feats

out = agent_api.build_features(make_features)
print("built:", out.shape)
print(out.groupby('snapshot_day').size())
base = agent_api.load_saved("e002_mix.parquet")
m = base.merge(out, on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape, "| null rows:", m.isna().all(axis=1).sum())
path = agent_api.save_table(m, "e003_mktg.parquet")
print(path)
