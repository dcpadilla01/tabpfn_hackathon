import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def spline_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value']]
    hh = pd.Index(view.households if isinstance(view.households, pd.Index) else np.asarray(view.households).ravel())
    out = pd.DataFrame(index=hh)
    s28 = tx[tx.day > day-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    s84 = tx[tx.day > day-84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    s364 = tx[tx.day > day-364].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    rec = (day - tx.groupby('household_key')['day'].max()).reindex(hh).fillna(999)
    # EWMA-style fwd mean proxy: use E008's fwd28_mean? Not available inside; rebuild as wk-based EWMA
    wks = [tx[(tx.day > day-7*(k+1)) & (tx.day <= day-7*k)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values for k in range(8)]
    wks = np.array(wks)  # 8 x n
    ewma = np.zeros(len(hh)); w = 0.0
    for k in range(8):
        w = 0.7*w + (1 if k==0 else 0.3)
        ewma += w*wks[k]
    ewma = ewma / w
    out['ewma8'] = ewma
    out['L_s28'] = np.log1p(s28.values); out['L_s84'] = np.log1p(s84.values)
    out['L_s364'] = np.log1p(s364.values); out['L_ewma'] = np.log1p(np.clip(ewma,0,None))
    out['rec'] = rec.values
    # linear-spline bases (hinge functions) at fixed quantile knots of log-spend
    def hinges(col, nknots=8):
        x = out[col].values
        qs = np.quantile(x, np.linspace(0.05, 0.95, nknots))
        qs = np.unique(qs)
        for j,q in enumerate(qs):
            out['h_%s_%d'%(col,j)] = np.maximum(x-q, 0)
    for c in ['L_s28','L_s84','L_s364','L_ewma']:
        hinges(c, 8)
    # recency hinges (on raw days, capped)
    r = np.minimum(out['rec'].values, 200.0)
    for q in [3,7,10,14,21,28,42,56,84,112,150]:
        out['hr_%d'%q] = np.maximum(r-q, 0)
    # two-part lookup: P(y>0) and E[y|y>0] by (s28 decile x recency bucket) from past anchors
    rows_p, rows_l, keys_all = [], [], []
    anchors = [day-28*k for k in range(1,13) if day-28*k >= 56]
    def keyfn(s28v, recv, qs, rb_edges):
        qb = pd.Series(np.searchsorted(qs, np.log1p(np.clip(s28v,0,None)))).astype(str)
        rb = pd.Series(np.searchsorted(rb_edges, recv)).astype(str)
        return (qb+'_'+rb).values
    ps, ls = [], []
    for a in anchors:
        t = tx[tx.day <= a]
        sa = t[t.day > a-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        ra = (a - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999).values
        ya = t[(t.day > a) & (t.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        ps.append((ya>0).astype(float)); ls.append(ya)
        ps_key = None
    # build lookup from pooled anchors
    allkeys = []
    for i,a in enumerate(anchors):
        t = tx[tx.day <= a]
        sa = t[t.day > a-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        ra = (a - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999).values
        ya = t[(t.day > a) & (t.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        qs = np.quantile(np.log1p(sa), np.linspace(0,1,11))[1:-1]
        rbe = np.array([7,14,28,56,112])
        allkeys.append(pd.DataFrame({'k':keyfn(sa,ra,qs,rbe),'p':(ya>0).astype(float),'l':ya}))
    if allkeys:
        pool = pd.concat(allkeys)
        tab_p = pool.groupby('k')['p'].mean(); tab_l = pool.groupby('k')['l'].mean()
        gp = pool['p'].mean(); gl = pool['l'].mean()
        qs_cur = np.quantile(np.log1p(s28.values), np.linspace(0,1,11))[1:-1]
        kc = keyfn(s28.values, rec.values, qs_cur, np.array([7,14,28,56,112]))
        p_hat = pd.Series(kc).map(tab_p).astype(float).fillna(gp).values
        l_hat = pd.Series(kc).map(tab_l).astype(float).fillna(gl).values
        out['p_active'] = p_hat; out['lvl_given_active'] = l_hat
        out['pred_2p'] = p_hat * l_hat
        out['L_pred2p'] = np.log1p(out['pred_2p'].clip(lower=0))
    return out.astype(float)

bf = agent_api.build_features(spline_feats)
print('built', bf.shape, bf.columns.tolist()[:8], '...')
e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
comb2 = comb.merge(e002.drop(columns=[c for c in e002.columns if c in comb.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('comb', comb.shape, 'comb2', comb2.shape)

tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def loo_mae(table, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    maes = []
    for d in tr_days:
        tr = m[m.snapshot_day != d]; va = m[m.snapshot_day == d]
        Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
        ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
        mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
        w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        p = B@w
        maes.append(np.abs(p-yva).mean())
    return np.mean(maes), maes

for name, tab in [('E008', e008), ('E008+spline2p', comb), ('E008+spline2p+mkt', comb2)]:
    for a in ([1.0, 100.0] if name!='E008' else [1.0, 100.0]):
        lm, _ = loo_mae(tab, alpha=a)
        print('LOO proxy %-18s alpha=%g: %.3f' % (name, a, lm))
path = agent_api.save_table(comb, 'e009_spline2p.parquet'); print('saved', path)