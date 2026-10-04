e = load_saved('e011_price.parquet')
t = train_targets()
keys = ['household_key','snapshot_day']
e = e.sort_values(keys).reset_index(drop=True)

def sp_b(s):
    s = s.fillna(-1.0)
    return np.digitize(s, [0.01, 20, 50, 100, 175, 300])
def rec_b(s):
    s = s.fillna(999.0)
    return np.digitize(s, [8, 15, 29, 57])
def tr_b(s):
    s = s.fillna(0.0)
    return np.digitize(s, [1, 4, 7, 11])
def sp_coarse(s):
    s = s.fillna(-1.0)
    return np.digitize(s, [0.01, 50, 150])
def rec_coarse(s):
    s = s.fillna(999.0)
    return np.digitize(s, [8, 29])

e['sp_b'] = sp_b(e['spend28']); e['rec_b'] = rec_b(e['recency'])
e['tr_b'] = tr_b(e['trips28']); e['lg_b'] = sp_b(e['lag_spend_1'])
e['cross_b'] = sp_coarse(e['spend28'])*10 + rec_coarse(e['recency'])
bcols = ['sp_b','rec_b','tr_b','lg_b','cross_b']
tt = t.merge(e[keys+bcols], on=keys, how='left')

K_MEAN, K_ZERO = 30.0, 50.0
te_cols = ['te_sp28','te_sp28z','te_rec','te_trip','te_lag1','te_cross','prior_mean']
TE = np.full((len(e), len(te_cols)), np.nan)
fam = [('sp_b','te_sp28','te_sp28z'), ('rec_b','te_rec',None), ('tr_b','te_trip',None),
       ('lg_b','te_lag1',None), ('cross_b','te_cross',None)]
ycol = TARGET
for d in sorted(e['snapshot_day'].unique()):
    idx = (e['snapshot_day']==d).values
    hist = tt[tt['snapshot_day'] < d]
    if len(hist)==0:
        continue
    prior = float(hist[ycol].mean()); pz = float((hist[ycol]==0).mean())
    TE[idx, te_cols.index('prior_mean')] = prior
    for bcol, mcol, zcol in fam:
        g = hist.groupby(bcol)[ycol]
        n = g.size(); s = g.sum(); z = g.apply(lambda v:(v==0).sum())
        gm = (s + K_MEAN*prior)/(n + K_MEAN)
        rb = e.loc[idx, bcol]
        TE[idx, te_cols.index(mcol)] = rb.map(gm).values
        if zcol:
            gz = (z + K_ZERO*pz)/(n + K_ZERO)
            TE[idx, te_cols.index(zcol)] = rb.map(gz).values
te_df = pd.DataFrame(TE, columns=te_cols)
out = pd.concat([e, te_df], axis=1)
p = save_table(out, 'e013_te_clean.parquet')
print('saved', p, out.shape)