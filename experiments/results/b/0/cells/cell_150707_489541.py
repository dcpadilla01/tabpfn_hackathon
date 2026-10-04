
import pandas as pd, numpy as np

def new_feats(view, sd):
    t = view.transactions
    pr = view.products[['product_id','commodity_desc','brand','manufacturer']].drop_duplicates('product_id')
    m = t.merge(pr, on='product_id', how='left')
    m = m[m.day > sd-112]
    tot = m.groupby('household_key')['sales_value'].sum()

    cs = m.groupby(['household_key','commodity_desc'], observed=True)['sales_value'].sum().rename('s').reset_index()
    cs = cs[cs['s']>0]
    cs['rank'] = cs.groupby('household_key')['s'].rank(ascending=False, method='first')
    top1 = cs[cs['rank']==1].set_index('household_key')['s']
    top3s = cs[cs['rank']<=3].groupby('household_key')['s'].sum()
    top5s = cs[cs['rank']<=5].groupby('household_key')['s'].sum()
    ncom = cs.groupby('household_key').size()

    bs = m.groupby(['household_key','brand'], observed=True)['sales_value'].sum().unstack(fill_value=0.0)
    priv_share = (bs.get('Private',0.0) / bs.sum(axis=1).replace(0,np.nan))

    mu = m[m['quantity']>0].copy()
    mu['up'] = mu['sales_value']/mu['quantity']
    mu = mu[mu['up']<100]
    up_mean = mu.groupby('household_key')['up'].mean()
    up_med  = mu.groupby('household_key')['up'].median()
    up_p90  = mu.groupby('household_key')['up'].quantile(0.9)
    cheap_u = mu[mu['up']<1].groupby('household_key')['quantity'].sum()
    prem_u  = mu[mu['up']>=5].groupby('household_key')['quantity'].sum()
    units = mu.groupby('household_key')['quantity'].sum().replace(0,np.nan)

    ms = m.groupby(['household_key','manufacturer'], observed=True)['sales_value'].sum().reset_index()
    ms['rank'] = ms.groupby('household_key')['sales_value'].rank(ascending=False, method='first')
    topm = ms[ms['rank']==1].set_index('household_key')['sales_value']
    nman = ms.groupby('household_key').size()

    m2 = m[m.day > sd-56]
    m1 = m[(m.day > sd-112) & (m.day <= sd-56)]
    def topset(mm):
        c = mm.groupby(['household_key','commodity_desc'], observed=True)['sales_value'].sum().reset_index()
        c = c[c['sales_value']>0]
        c['r'] = c.groupby('household_key')['sales_value'].rank(ascending=False, method='first')
        c = c[c['r']<=8]
        return c.groupby('household_key')['commodity_desc'].apply(lambda s: set(s))
    s1, s2 = topset(m1), topset(m2)
    j = pd.concat([s1.rename('a'), s2.rename('b')], axis=1).dropna()
    jac = pd.Series([len(a&b)/max(1,len(a|b)) for a,b in zip(j['a'],j['b'])], index=j.index)

    # persistence of dominant commodity across the two 56d halves
    t1_2 = m2.groupby(['household_key','commodity_desc'], observed=True)['sales_value'].sum()
    top1_recent = t1_2.groupby('household_key', observed=True).max()
    top1_persist = (top1_recent/top1).clip(0,1)

    return pd.DataFrame({
        'top1_com_share_112': top1/tot,
        'top1_com_spend_112': top1,
        'top3_com_share_112': top3s/tot,
        'top5_com_share_112': top5s/tot,
        'ncom_112': ncom,
        'brand_priv_share_112': priv_share,
        'uprice_mean_112': up_mean,
        'uprice_med_112': up_med,
        'uprice_p90_112': up_p90,
        'cheap_units_share_112': cheap_u/units,
        'prem_units_share_112': prem_u/units,
        'topman_share_112': topm/tot,
        'nman_112': nman,
        'com_top_jaccard_56': jac,
        'top1_persist_112': top1_persist,
    })

base = agent_api.load_saved('e007_lagseq.parquet')
def build(view, sd):
    f = new_feats(view, sd)
    out = base.merge(f, left_on='household_key', right_index=True, how='left')
    return out.set_index('household_key').drop(columns=['snapshot_day'])

X = agent_api.build_features(build)
print(X.shape)
print(X[['top1_com_share_112','ncom_112','uprice_mean_112','com_top_jaccard_56','top1_persist_112']].describe().T.round(3))
path = agent_api.save_table(X.reset_index(), 'e009_basket.parquet')
print(path)
