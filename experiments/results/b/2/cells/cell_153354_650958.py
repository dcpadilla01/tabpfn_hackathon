import pandas as pd, numpy as np, agent_api

base = agent_api.load_saved('e009_demo.parquet')

def make_feats(view, snapshot_day):
    sd = snapshot_day
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    tx = tx.assign(
        d_any=(tx.retail_disc < 0) | (tx.coupon_disc < 0) | (tx.coupon_match_disc < 0),
        d_amt=(-(tx.retail_disc + tx.coupon_disc + tx.coupon_match_disc)).clip(lower=0),
        d_ret=(-tx.retail_disc).clip(lower=0),
        d_cpn=(-tx.coupon_disc).clip(lower=0),
    )
    out = pd.DataFrame(index=hh)
    for W in (28, 84, 364):
        t = tx[tx.day > sd - W]
        g = t.groupby('household_key')
        spend = g['sales_value'].sum()
        n = g.size()
        out[f'disc_item_share_{W}'] = (g['d_any'].sum() / n).reindex(hh).fillna(0.0)
        out[f'disc_amt_rate_{W}'] = (g['d_amt'].sum() / spend.replace(0, np.nan)).reindex(hh).fillna(0.0)
        out[f'retail_disc_rate_{W}'] = (g['d_ret'].sum() / spend.replace(0, np.nan)).reindex(hh).fillna(0.0)
    t84 = tx[tx.day > sd - 84]
    g84 = t84.groupby('household_key')
    sp84 = g84['sales_value'].sum().reindex(hh).fillna(0.0)
    out['coupon_disc_rate_84'] = (g84['d_cpn'].sum().reindex(hh).fillna(0.0) / sp84.replace(0, np.nan)).fillna(0.0)
    out['coupon_user_84'] = g84['d_cpn'].sum().reindex(hh).fillna(0.0).gt(0).astype(float)

    cr = view.coupon_redemptions
    if len(cr):
        gcr = cr.groupby('household_key')['day']
        out['red_total'] = np.log1p(gcr.size().reindex(hh).fillna(0.0))
        for W in (84, 168, 364):
            out[f'red_{W}'] = np.log1p(cr[cr.day > sd - W].groupby('household_key').size().reindex(hh).fillna(0.0))
        out['days_since_red'] = sd - gcr.max().reindex(hh)
    else:
        for c in ['red_total','red_84','red_168','red_364']:
            out[c] = 0.0
        out['days_since_red'] = np.nan

    camp = view.campaigns.rename(columns={'description': 'camp_desc'})
    ct = view.campaign_targets.merge(camp[['campaign','start_day','end_day']], on='campaign', how='left')
    gct = ct.groupby('household_key')
    out['tgt_total'] = np.log1p(gct['campaign'].nunique().reindex(hh).fillna(0.0))
    for tp in ('TypeA', 'TypeB', 'TypeC'):
        sub = ct[ct['description'].astype(str).str.startswith(tp)]
        out[f'tgt_{tp}'] = np.log1p(sub.groupby('household_key').size().reindex(hh).fillna(0.0))
    out['days_since_tgt_start'] = sd - gct['start_day'].max().reindex(hh)
    act = ct[(ct.start_day <= sd) & (ct.end_day >= sd)].groupby('household_key').size()
    out['tgt_active_now'] = act.reindex(hh).fillna(0.0).gt(0).astype(float)
    rec = ct[ct.start_day > sd - 84].groupby('household_key').size()
    out['tgt_recent84'] = np.log1p(rec.reindex(hh).fillna(0.0))
    return out

feats = agent_api.build_features(make_feats)
print('feats', feats.shape)

df = feats.reset_index().merge(base, on=['household_key', 'snapshot_day'], how='inner')
print('merged', df.shape)
assert len(df) == len(base)

df['ix_disc84_income'] = df['disc_item_share_84'] * df['income_ord']
df['ix_disc84_size'] = df['disc_item_share_84'] * df['size_ord']
df['ix_red84_income'] = df['red_84'] * df['income_ord']

newcols = [c for c in feats.columns if c not in ('household_key','snapshot_day')] + ['ix_disc84_income','ix_disc84_size','ix_red84_income']
print('n new feats:', len(newcols), 'total cols:', df.shape[1])
print(df[newcols].describe().T[['mean','std','min','max']].to_string())
path = agent_api.save_table(df, 'e011_discounts.parquet')
print(path)