import pandas as pd, numpy as np
import agent_api as api

def fn(view, day):
    hh = pd.Index(view.households)
    out = pd.DataFrame(index=hh)

    camps = view.campaigns[['campaign','start_day','end_day']]
    ct = view.campaign_targets.merge(camps, on='campaign', how='left')
    desc = ct['description'].astype(object).where(ct['description'].notna(), 'NA').astype(str)
    ct['description'] = desc
    piv = pd.crosstab(ct['household_key'], ct['description'])
    for c in ['TypeA','TypeB','TypeC']:
        out['m_tgt_'+c] = piv[c] if c in piv.columns else 0
    out['m_tgt_any'] = ct.groupby('household_key').size()
    act = ct[(ct.start_day<=day)&(ct.end_day>=day)]
    out['m_tgt_active'] = act.groupby('household_key').size()
    ls = ct.groupby('household_key')['start_day'].max()
    out['m_days_since_camp'] = (day - ls).astype(float)

    cr = view.coupon_redemptions
    for w in (28, 84, 364):
        out['m_red_%d'%w] = cr[cr.day > day-w].groupby('household_key').size()
    lr = cr.groupby('household_key')['day'].max()
    out['m_days_since_red'] = (day - lr).astype(float)
    out['m_n_camp_red'] = cr.groupby('household_key')['campaign'].nunique()

    txw = view.transactions
    txw = txw[txw.day > day-28][['household_key','basket_id','product_id','store_id','week_no','sales_value']]
    dm = view.display_mailer[['product_id','store_id','week_no','display','mailer']]
    mg = txw.merge(dm, on=['product_id','store_id','week_no'], how='left')
    disp = pd.to_numeric(mg['display'], errors='coerce').fillna(0)
    mg['disp_flag'] = (disp > 0).astype(float)
    mailer_s = mg['mailer'].astype(object).where(mg['mailer'].notna(), '0').astype(str)
    mg['mail_flag'] = (mailer_s != '0').astype(float)
    mg['sv'] = mg['sales_value'].abs()
    mg['sv_disp'] = mg['sv']*mg['disp_flag']
    mg['sv_mail'] = mg['sv']*mg['mail_flag']
    g = mg.groupby('household_key')
    sv = g['sv'].sum()
    out['m_spend28'] = sv
    out['m_frac_disp'] = g['sv_disp'].sum() / sv.replace(0, np.nan)
    out['m_frac_mail'] = g['sv_mail'].sum() / sv.replace(0, np.nan)
    out['m_mean_disp'] = g['disp_flag'].mean()
    mrows = mg[mg.mail_flag > 0]
    out['m_mail_letters'] = mrows.groupby('household_key')['mailer'].nunique()
    out['m_baskets_mail'] = mrows.groupby('household_key')['basket_id'].nunique()
    out['m_baskets28'] = g['basket_id'].nunique()

    out = out.reindex(hh)
    for c in out.columns:
        if out[c].dtype.kind in 'if' and c.startswith(('m_red','m_tgt','m_mail','m_baskets','m_n_')):
            out[c] = out[c].fillna(0)
    return out

df = api.build_features(fn)
print(df.shape)
print(df.head(3))
print(df.isna().mean().round(3).to_dict())
path = api.save_table(df, 'mkt_v1.parquet')
print(path)
