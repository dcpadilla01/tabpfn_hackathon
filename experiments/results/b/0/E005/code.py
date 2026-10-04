import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('season.parquet')
print("E004 table:", t.shape)
print(list(t.columns))
tt = agent_api.train_targets()
print("\ntargets:", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())

v = agent_api.snapshot()
print("\ncampaigns:", v.campaigns.shape); print(v.campaigns.head(8))
print(v.campaigns.description.value_counts())
print("\ncampaign_targets:", v.campaign_targets.shape)
print(v.campaign_targets.description.value_counts())
print("hh per campaign sample:")
print(v.campaign_targets.groupby('campaign').size().head(10))
print("distinct hh targeted:", v.campaign_targets.household_key.nunique())
print("\ncoupon_redemptions:", v.coupon_redemptions.shape)
print(v.coupon_redemptions.head())
print("redemptions per hh describe:")
print(v.coupon_redemptions.groupby('household_key').size().describe())
print("\ncoupons:", v.coupons.shape); print(v.coupons.head())
print("\ndisplay_mailer:", v.display_mailer.shape); print(v.display_mailer.head())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def probe(view, day):
    hh = view.households
    print(f"day={day} week={view.week} n_hh={len(hh)}",
          "tx:", view.transactions.shape,
          "camp:", view.campaigns.shape,
          "ct:", view.campaign_targets.shape,
          "red:", view.coupon_redemptions.shape,
          "dm:", view.display_mailer.shape,
          "prods:", view.products.shape)
    print("camp desc:", view.campaigns.description.value_counts().to_dict())
    print("ct desc:", view.campaign_targets.description.value_counts().to_dict())
    print("ct hh:", view.campaign_targets.household_key.nunique())
    return pd.DataFrame({'x': 1.0}, index=pd.Index(hh, name='household_key'))

bf = agent_api.build_features(probe)
print(bf.shape)
print(bf.head())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('season.parquet')
print('base', base.shape)

def fn(view, day):
    hh = pd.Index(view.households, name='household_key')
    out = pd.DataFrame(index=hh)
    tx = view.transactions
    camps = view.campaigns
    act_camps = set(camps.loc[camps['end_day'] >= day, 'campaign']) if len(camps) else set()

    # ---- campaign targeting (campaigns already started) ----
    ct = view.campaign_targets
    if len(ct) and len(camps):
        cinfo = camps.set_index('campaign')[['start_day', 'end_day']]
        ct2 = ct.merge(cinfo, left_on='campaign', right_index=True, how='left')
        g = ct2.groupby('household_key')
        out['n_tgt_ever'] = g.size().reindex(hh).fillna(0.0)
        sd = g['start_day'].agg(['min', 'max']).reindex(hh)
        out['days_since_first_tgt'] = day - sd['min']
        out['days_since_last_tgt'] = day - sd['max']
        act = ct2[ct2['end_day'] >= day]
        ga = act.groupby('household_key')
        out['n_tgt_active'] = ga.size().reindex(hh).fillna(0.0)
        out['tgt_active_remaining'] = (act['end_day'] - day).groupby(act['household_key']).mean().reindex(hh).fillna(0.0)
        for t in ('TypeA', 'TypeB', 'TypeC'):
            out['t' + t[-1] + '_lt'] = ct2[ct2['description'] == t].groupby('household_key').size().reindex(hh).fillna(0.0)
            out['t' + t[-1] + '_act'] = act[act['description'] == t].groupby('household_key').size().reindex(hh).fillna(0.0)
    else:
        for c in ['n_tgt_ever','n_tgt_active','tgt_active_remaining','tA_lt','tB_lt','tC_lt','tA_act','tB_act','tC_act']:
            out[c] = 0.0
        out['days_since_first_tgt'] = np.nan
        out['days_since_last_tgt'] = np.nan
    out['targeted_flag'] = (out['n_tgt_ever'] > 0).astype(float)

    # ---- coupon redemption engagement ----
    r = view.coupon_redemptions
    if len(r):
        g = r.groupby('household_key')
        out['red_lt'] = g.size().reindex(hh).fillna(0.0)
        out['red28'] = r[r['day'] > day - 28].groupby('household_key').size().reindex(hh).fillna(0.0)
        out['red112'] = r[r['day'] > day - 112].groupby('household_key').size().reindex(hh).fillna(0.0)
        out['days_since_last_red'] = day - g['day'].max().reindex(hh)
        out['red_ncamp'] = g['campaign'].nunique().reindex(hh).fillna(0.0)
        if act_camps:
            out['red_active'] = r[r['campaign'].isin(act_camps)].groupby('household_key').size().reindex(hh).fillna(0.0)
        else:
            out['red_active'] = 0.0
    else:
        for c in ['red_lt','red28','red112','red_ncamp','red_active']:
            out[c] = 0.0
        out['days_since_last_red'] = np.nan
    out['red_rate'] = out['red_lt'] / out['n_tgt_ever'].replace(0.0, np.nan)

    # ---- 28d spend on products covered by active campaigns' coupons ----
    m28 = tx[(tx['day'] > day - 28) & (tx['day'] <= day)]
    cps = view.coupons
    if len(cps) and act_camps:
        pids = cps.loc[cps['campaign'].isin(act_camps), 'product_id'].unique()
        mm = m28[m28['product_id'].isin(pids)]
        gsp = mm.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        out['cpn_prod_spend28'] = gsp
        s28 = m28.groupby('household_key').size().reindex(hh).fillna(0.0)
        out['cpn_prod_share28'] = gsp / 1.0
        out['cpn_prod_share28'] = np.where(s28.values > 0, gsp.values / np.maximum(s28.values, 1.0), np.nan)
    else:
        out['cpn_prod_spend28'] = 0.0
        out['cpn_prod_share28'] = np.nan

    # ---- demographics (ordinal encodings) ----
    dm = view.demographics
    if dm is not None and len(dm):
        dmi = dm.set_index('household_key')
        out['demo_age'] = dmi['classification_1'].str.extract(r'(\d+)')[0].astype(float).reindex(hh)
        out['demo_class3'] = dmi['classification_3'].str.extract(r'(\d+)')[0].astype(float).reindex(hh)
        out['demo_hhsize'] = pd.to_numeric(dmi['classification_4'].replace('5+', '5'), errors='coerce').reindex(hh)
        out['demo_kids'] = dmi['kid_category_desc'].map({'None/Unknown': 0.0, '1': 1.0, '2': 2.0, '3+': 3.0}).reindex(hh)
        out['demo_homeowner'] = dmi['homeowner_desc'].reindex(hh).astype(object)
        out['demo_c2'] = dmi['classification_2'].reindex(hh).astype(object)
        out['demo_c5'] = dmi['classification_5'].reindex(hh).astype(object)
        out['has_demo'] = out['demo_age'].notna().astype(float)
    return out

newf = agent_api.build_features(fn)
print('newf', newf.shape)
print(newf.drop(columns=['household_key','snapshot_day']).mean(numeric_only=True).round(3).to_string())
print('targeted_flag mean:', newf['targeted_flag'].mean().round(4))
print('red_lt>0 share:', (newf['red_lt']>0).mean().round(4))
print('has_demo mean:', newf['has_demo'].mean().round(4) if 'has_demo' in newf else 'MISSING')

overlap = [c for c in newf.columns if c in base.columns and c not in ('household_key','snapshot_day')]
print('overlapping cols with base:', overlap)
merged = base.merge(newf.drop(columns=overlap), on=['household_key','snapshot_day'], how='left')
print('merged', merged.shape)
agent_api.save_table(merged, 'mkt_demo.parquet')
print('saved')