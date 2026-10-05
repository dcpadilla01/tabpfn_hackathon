
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def fn(view, sd):
    hh = view.households
    tx = view.table('transactions'); d = tx['day']
    f = pd.DataFrame(index=hh); f.index.name = 'household_key'
    def sp(w): return tx[d>sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    def tr(w): return tx[d>sd-w].groupby('household_key')['basket_id'].nunique().reindex(hh, fill_value=0.0)
    def pr(w): return tx[d>sd-w].groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0.0)
    def na(w): return tx[d>sd-w].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)
    s28,s56,s84,s168,s364 = sp(28),sp(56),sp(84),sp(168),sp(364)
    t28,t84 = tr(28),tr(84); p28,p84 = pr(28),pr(84)
    # momentum / ratios / interactions
    f['m_sp28_56'] = s28 - 0.5*s56
    f['m_sp56_84'] = s56 - 0.5*s84
    f['r_sp28_84'] = s28/(s84+1.0)
    f['r_sp84_364'] = s84/(s364+1.0)
    f['r_tr28_84'] = t28/(t84+1.0)
    f['r_pr28_84'] = p28/(p84+1.0)
    f['spt_84'] = s84/(t84+1.0)
    f['spd_84'] = s84/(na(84)+1.0)
    f['i_sp28_rise'] = s28*s28/(s56+1.0)
    f['i_sp84_rise'] = s84*s84/(s364+1.0)
    # store loyalty share (168d)
    t3 = tx[d>sd-168]
    ss = t3.groupby(['household_key','store_id'])['sales_value'].sum()
    f['topstore'] = ss.groupby('household_key').max().reindex(hh, fill_value=0.0)/(s168+1e-9)
    # weekly volatility (12w)
    t2 = tx[d>sd-84].copy(); t2['wk'] = t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key')
    f['wkstd'] = ws.std().reindex(hh, fill_value=0.0)
    f['wkmax'] = ws.max().reindex(hh, fill_value=0.0)
    # seasonal lag + recent activity
    f['splag1y'] = tx[(d>sd-392)&(d<=sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['nact14'] = na(14)
    try:
        e2 = agent_api.load_saved('e002_mix.parquet')
        sub = e2[e2['snapshot_day']==sd].drop(columns=['snapshot_day']).set_index('household_key').reindex(hh)
        out = sub.join(f)
        return out
    except Exception as e:
        # fallback core
        g = tx.groupby('household_key')
        f['tenure'] = (sd - g['day'].min().reindex(hh)).clip(lower=0)
        f['days_since_last'] = sd - g['day'].max().reindex(hh)
        for w in (28,56,84,168,364,728):
            f[f'sp{w}'] = sp(w); f[f'trips{w}'] = tr(w); f[f'prods{w}'] = pr(w); f[f'nact{w}'] = na(w)
            f[f'stores{w}'] = tx[d>sd-w].groupby('household_key')['store_id'].nunique().reindex(hh, fill_value=0)
            f[f'qty{w}'] = tx[d>sd-w].groupby('household_key')['quantity'].sum().reindex(hh, fill_value=0.0)
            bs = tx[d>sd-w].groupby(['household_key','basket_id'])['sales_value'].sum()
            f[f'avgbs{w}'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
            f[f'maxbs{w}'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
        for w in (84,364):
            f[f'cdisc{w}'] = tx[d>sd-w].groupby('household_key')['coupon_disc'].sum().reindex(hh, fill_value=0.0)
            f[f'rdisc{w}'] = tx[d>sd-w].groupby('household_key')['retail_disc'].sum().reindex(hh, fill_value=0.0)
        return f

X = agent_api.build_features(fn)
print(X.shape)
print([c for c in X.columns if c.startswith(('m_','r_','i_','top','wk','splag','nact14','spt','spd'))])
print(X.head(3).iloc[:, :8])
p = agent_api.save_table(X, 'e003_momentum.parquet')
print(p)
