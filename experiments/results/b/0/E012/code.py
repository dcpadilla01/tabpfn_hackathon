import agent_api, pandas as pd, numpy as np

try:
    names = ['mkt_demo','rfm28','rich_behavioral','season','e006_composition','e007_lagseq','e010_decay','e011_price']
    base = None
    for n in names:
        t = agent_api.load_saved(n + '.parquet')
        if base is None:
            base = t.copy()
        else:
            dup = [c for c in t.columns if c in base.columns and c not in ('household_key','snapshot_day')]
            base = base.merge(t.drop(columns=dup), on=['household_key','snapshot_day'], how='left')
    print('base', base.shape, 'feats', base.shape[1]-2)

    snap_days = sorted(base.snapshot_day.unique())
    train_days = agent_api.snapshot_days()['train']
    print('snaps', snap_days)

    tx = agent_api.snapshot().transactions
    lastd_all = tx.groupby('household_key').day.max()

    ctx_rows = []
    for s in snap_days:
        sub = base[base.snapshot_day == s]
        hh = sub.household_key
        w = tx[(tx.day > s-28) & (tx.day <= s)]
        sp = w.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        tr = w.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0.0)
        rc = (s - lastd_all.reindex(hh)).astype(float)
        ctx_rows.append(pd.DataFrame({'household_key': hh.values, 'snapshot_day': s,
                                      'c_spend28': sp.values, 'c_trips28': tr.values, 'c_recency': rc.values}))
    base = base.merge(pd.concat(ctx_rows, ignore_index=True), on=['household_key','snapshot_day'], how='left')

    t_rows = []
    for s in train_days:
        sub = base[base.snapshot_day == s]
        hh = sub.household_key
        g = tx[(tx.day > s) & (tx.day <= s+28)].groupby('household_key').sales_value.sum()
        t_rows.append(pd.DataFrame({'household_key': hh.values, 'snapshot_day': s,
                                    'target': hh.map(g).fillna(0.0).values}))
    base = base.merge(pd.concat(t_rows, ignore_index=True), on=['household_key','snapshot_day'], how='left')
    print('target NaN (should equal n val rows):', base.target.isna().sum())

    _, e1 = pd.qcut(base.c_spend28, 4, retbins=True, duplicates='drop')
    _, e2 = pd.qcut(base.c_trips28, 3, retbins=True, duplicates='drop')
    _, e3 = pd.qcut(base.c_recency, 3, retbins=True, duplicates='drop')
    E_SP = np.concatenate(([-np.inf], e1[1:-1], [np.inf]))
    E_TR = np.concatenate(([-np.inf], e2[1:-1], [np.inf]))
    E_RC = np.concatenate(([-np.inf], e3[1:-1], [np.inf]))
    print('n buckets', len(E_SP)-1, len(E_TR)-1, len(E_RC)-1)

    def bstr(b):
        return b.map(lambda v: 'NA' if pd.isna(v) else str(int(v)))

    def keystrs(sp, tr, rc):
        s4 = bstr(pd.cut(sp, E_SP, labels=False))
        str_ = bstr(pd.cut(tr, E_TR, labels=False))
        src = bstr(pd.cut(rc, E_RC, labels=False))
        return s4, s4 + '|' + src, s4 + '|' + str_ + '|' + src

    k4, k12, k36 = keystrs(base.c_spend28, base.c_trips28, base.c_recency)
    base['k4'] = k4.values; base['k12'] = k12.values; base['k36'] = k36.values

    priors = {}
    for s in snap_days:
        pr = base[(base.snapshot_day < s) & base.target.notna()]
        if len(pr) == 0:
            priors[s] = None
            continue
        info = {'gm': float(pr.target.mean()), 'n': int(len(pr))}
        for nm, kc in [('b4','k4'), ('b12','k12'), ('b36','k36')]:
            g = pr.groupby(kc).target.agg(['mean','count'])
            info[nm] = (g['mean'].to_dict(), g['count'].to_dict())
        priors[s] = info
    print('prior sizes', {s: (priors[s]['n'] if priors[s] else 0) for s in snap_days})

    K = 20.0
    def fn(view, snapshot_day):
        s = snapshot_day
        txv = view.transactions
        h = view.households
        if isinstance(h, pd.Series) and h.index.name == 'household_key':
            idx = h.index
        elif isinstance(h, pd.Index):
            idx = h
        else:
            idx = pd.Index(np.asarray(h).ravel(), name='household_key')
        w = txv[(txv.day > s-28) & (txv.day <= s)]
        sp = w.groupby('household_key').sales_value.sum().reindex(idx).fillna(0.0)
        tr = w.groupby('household_key').basket_id.nunique().reindex(idx).fillna(0.0)
        rc = (s - txv.groupby('household_key').day.max().reindex(idx)).astype(float)
        kk4, kk12, kk36 = keystrs(sp, tr, rc)
        out = pd.DataFrame(index=idx)
        info = priors[s]
        if info is None:
            for nm in ('b4','b12','b36'):
                out['x_'+nm] = np.nan; out['x_'+nm+'_cnt'] = np.nan
            out['x_gm'] = np.nan; out['x_gn'] = np.nan
            return out
        gm, gn = info['gm'], info['n']
        for nm, ks in (('b4',kk4), ('b12',kk12), ('b36',kk36)):
            m, c = info[nm]
            cv = ks.map(c).fillna(0.0)
            mv = ks.map(m).fillna(gm)
            out['x_'+nm] = (cv*mv + K*gm) / (cv + K)
            out['x_'+nm+'_cnt'] = np.log1p(cv)
        out['x_gm'] = gm
        out['x_gn'] = float(gn)
        return out

    enc = agent_api.build_features(fn)
    if 'household_key' not in enc.columns:
        enc = enc.reset_index()
    print('enc', enc.shape)

    full = base.drop(columns=['k4','k12','k36']).merge(enc, on=['household_key','snapshot_day'], how='left')
    print('full', full.shape, 'feats', full.shape[1]-2)

    v = full[full.snapshot_day == 431]
    print('corr@431 x_b36/t %.3f  x_b12/t %.3f  c_spend28/t %.3f' % (
        np.corrcoef(v.x_b36, v.target)[0,1], np.corrcoef(v.x_b12, v.target)[0,1],
        np.corrcoef(v.c_spend28, v.target)[0,1]))
    print('val coverage', full[full.snapshot_day==459][['x_b36','x_b12','x_b4','x_gm']].notna().mean().to_dict())

    full = full.drop(columns=['target'])
    path = agent_api.save_table(full, 'e012_xenc')
    print('saved', path)
except Exception as e:
    print('ERR:', repr(e))