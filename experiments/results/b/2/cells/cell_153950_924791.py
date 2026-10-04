def fn(view, sd):
    t = view.transactions
    idx = pd.Index(view.households, name='household_key')
    out = pd.DataFrame(index=idx)
    g = t.groupby(['household_key', 'basket_id'])
    b = pd.DataFrame({
        'bval': g['sales_value'].sum(),
        'bday': g['day'].min(),
        'blines': g['product_id'].nunique(),
        'bqty': g['quantity'].sum(),
        'btime': g['trans_time'].max(),
    }).reset_index(level=1, drop=True)
    b28 = b[b['bday'] > sd - 28]
    b84 = b[b['bday'] > sd - 84]
    lastb = b28.sort_values('bday').groupby(level=0).tail(1)
    out['last_bval'] = lastb['bval']
    out['last_blines'] = lastb['blines']
    out['last_bqty'] = lastb['bqty']
    m84 = b84.groupby(level=0)['bval']
    out['bmean84'] = m84.mean()
    out['bstd84'] = m84.std()
    out['bmax84'] = m84.max()
    out['bmax28'] = b28.groupby(level=0)['bval'].max()
    out['trips28b'] = b28.groupby(level=0).size()
    out['trips84b'] = m84.size()
    t28 = t[t['day'] > sd - 28]
    out['qty28'] = t28.groupby('household_key')['quantity'].sum()
    out['spend28b'] = t28.groupby('household_key')['sales_value'].sum()
    dow = b84['bday'] % 7
    out['weekend_share84'] = b84.assign(w=(dow >= 5).astype(int)).groupby(level=0)['w'].mean()
    tt = b84['btime']
    out['evening_share84'] = b84.assign(e=(tt >= 1700).astype(int)).groupby(level=0)['e'].mean()
    out['morning_share84'] = b84.assign(m=(tt < 1200).astype(int)).groupby(level=0)['m'].mean()
    out['last_ratio'] = out['last_bval'] / out['bmean84']
    out['stockup28'] = out['bmax28'] / out['bmean84']
    out['bcv84'] = out['bstd84'] / out['bmean84']
    out['unit_price28'] = out['spend28b'] / out['qty28'].replace(0, np.nan)
    out['log_last_bval'] = np.log1p(out['last_bval'])
    return out

feats = build_features(fn)
print(feats.shape)
print(feats.drop(columns=['household_key','snapshot_day']).describe().T[['mean','std','min','max']].round(2).to_string())
