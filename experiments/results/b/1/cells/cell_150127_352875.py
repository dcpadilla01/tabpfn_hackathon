import pandas as pd, numpy as np, agent_api

def make_feats(view, day):
    t = view.transactions
    hh = pd.Index(view.households)
    g = t.groupby('household_key')
    # ---------- NEW: temporal-shape features ----------
    nf = pd.DataFrame(index=hh)
    for k in range(1,14):
        lo = day-28*k
        nf[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    age = day - t.day
    for hl in [28,56]:
        nf[f'ew{hl}'] = (t.sales_value*(0.5**(age/hl))).groupby(t.household_key).sum()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique())).reindex(hh)
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    nf['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else 0.0).astype(float)
    nf['n_gap21_182'] = g182.apply(lambda a: float((a>21).sum()))
    g84 = gaps.apply(lambda a: a[a<=84] if len(a)>0 else np.array([]))
    nf['n_gap14_84'] = g84.apply(lambda a: float((a>14).sum()))
    nf['gap_med'] = gaps.apply(lambda a: float(np.median(a)) if len(a)>0 else np.nan)
    W = nf[[f'w{i}' for i in range(1,8)]]
    Wm = W.mean(axis=1); Ws = W.std(axis=1)
    nf['w_std7'] = Ws; nf['w_max7'] = W.max(axis=1); nf['w_min7'] = W.min(axis=1)
    nf['w_cv7'] = np.where(Wm>0, Ws/Wm.replace(0,np.nan), 0.0)
    med = W[[f'w{i}' for i in range(2,8)]].median(axis=1)
    nf['spike'] = np.where(med>0, nf['w1']/med.replace(0,np.nan), np.where(nf['w1']>0, 10.0, 0.0))
    for c in [f'w{k}' for k in range(1,14)]+['ew28','ew56','gap_max182','n_gap21_182','n_gap14_84']:
        nf[c] = nf[c].fillna(0.0)
    # ---------- BASE: E001 ----------
    try:
        base = agent_api.load_saved('e001_history.parquet')
        if 'future_spend_4w' in base.columns: base = base.drop(columns=['future_spend_4w'])
        b = base[base['snapshot_day']==day].drop(columns=['snapshot_day']).set_index('household_key')
        res = b.join(nf, how='left')
    except Exception as e:
        print('load_saved inside fn failed:', e)
        d = t.day
        ef = pd.DataFrame(index=hh)
        ef['spend_7']  = t[d>day-7].groupby('household_key').sales_value.sum()
        ef['spend_28'] = nf['w1']
        ef['spend_56'] = t[d>day-56].groupby('household_key').sales_value.sum()
        ef['spend_84'] = t[d>day-84].groupby('household_key').sales_value.sum()
        ef['spend_182']= t[d>day-182].groupby('household_key').sales_value.sum()
        ef['spend_365']= t[d>day-365].groupby('household_key').sales_value.sum()
        ef['spend_all']= g.sales_value.sum()
        ef['nb_28'] = t[d>day-28].groupby('household_key').basket_id.nunique()
        ef['nb_56'] = t[d>day-56].groupby('household_key').basket_id.nunique()
        ef['nb_all'] = g.basket_id.nunique()
        ef['avg_basket_28'] = ef['spend_28']/ef['nb_28'].replace(0,np.nan)
        ef['days_since_last'] = day - g.day.max()
        ef['spend_prev28'] = nf['w2']
        ef['trend28'] = ef['spend_28'] - ef['spend_prev28']
        ef['qty_28'] = t[d>day-28].groupby('household_key').quantity.sum()
        ef['nprod_28'] = t[d>day-28].groupby('household_key').product_id.nunique()
        ef['spend_ly28'] = t[(d>day-364)&(d<=day-336)].groupby('household_key').sales_value.sum()
        fp = g.day.min()
        ef['ly_avail'] = (fp<=day-364).astype(float)
        ef['log_spend_28'] = np.log1p(ef['spend_28'].fillna(0))
        ef['log_spend_all'] = np.log1p(ef['spend_all'].fillna(0))
        ef['weekly_rate_84'] = ef['spend_84'].fillna(0)/84*7
        ef = ef.fillna({'spend_7':0,'spend_56':0,'spend_84':0,'spend_182':0,'spend_365':0,'spend_all':0,
                        'nb_28':0,'nb_56':0,'nb_all':0,'days_since_last':999,'spend_prev28':0,'trend28':0,
                        'qty_28':0,'nprod_28':0,'spend_ly28':0,'ly_avail':0})
        demo = view.demographics if hasattr(view,'demographics') else None
        ef['has_demographics'] = ef.index.isin(set(demo.household_key)).astype(float) if demo is not None else 0.0
        if demo is not None: ef = ef.join(demo.set_index('household_key').drop(columns=['household_key']), how='left')
        ef['snapshot_day_index'] = float(day)
        ef['week_of_year'] = float(((day+8)//7) % 52)
        res = ef.join(nf, how='left')
    return res

T = agent_api.build_features(make_feats)
print('built shape:', T.shape)
print('cols:', T.columns.tolist())
print('rows per snapshot:'); print(T.groupby('snapshot_day').size())
print('any all-NaN new cols:', [c for c in ['w8','w13','ew28','gap_max182','spike','w_cv7'] if T[c].notna().sum()==0])
path = agent_api.save_table(T, 'e006_seq_gaps')
print('saved:', path)