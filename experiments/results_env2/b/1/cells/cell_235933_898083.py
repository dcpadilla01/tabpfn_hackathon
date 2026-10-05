
import agent_api as A, pandas as pd, numpy as np, time
t0=time.time()
NEW = ['pm_share84','pm_share28','pc_share84','pc_share28','pdisp_share84','pmail_share84','pm_spend84','pm_spend28','p_wks84','snap_idx']

def build(view, s):
    wk = (s + 8) // 7
    tx = view.table('transactions')
    hh = view.households
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        hh = first[first <= s - 84].index
    if isinstance(hh, pd.DataFrame): hh = hh.index
    hh = pd.Index(hh)
    out = pd.DataFrame(index=hh)
    out['snap_idx'] = (s - 95) / 28.0
    t = tx[tx['day'] > s - 84]
    if len(t) == 0:
        for c in NEW[:-1]: out[c] = 0.0
        return out
    t = t[['household_key','day','product_id','store_id','sales_value']].copy()
    t['week_no'] = ((t['day'] + 8) // 7).astype('int16')
    dm = view.table('display_mailer')
    dm = dm[dm['week_no'] >= wk - 11]
    disp = pd.to_numeric(dm['display'].astype(str), errors='coerce').fillna(0).values
    mail = dm['mailer'].astype(str).values
    is_p = (disp > 0) | (mail != '0')
    pr = pd.DataFrame({'product_id': dm['product_id'].values[is_p],
                       'store_id': dm['store_id'].values[is_p],
                       'week_no': dm['week_no'].values[is_p],
                       'df': (disp[is_p] > 0).astype(np.float32),
                       'mf': np.isin(mail[is_p], ['A','D','H','F']).astype(np.float32)})
    pr = pr.groupby(['product_id','store_id','week_no'], as_index=False).max()
    p = view.table('products')[['product_id','commodity_desc']]
    pc = pr[['product_id','week_no']].drop_duplicates().merge(p, on='product_id', how='left')
    ci = pc.groupby(['commodity_desc','week_no'], as_index=False).size().rename(columns={'size':'npr'})
    tm = t.merge(pr, on=['product_id','store_id','week_no'], how='left')
    tm[['df','mf']] = tm[['df','mf']].fillna(0.0)
    tm['f'] = ((tm['df'] + tm['mf']) > 0).astype(np.float32)
    tc = t.merge(p, on='product_id', how='left').merge(ci, on=['commodity_desc','week_no'], how='left')
    tc['cf'] = tc['npr'].notna().astype(np.float32)
    def S(df, mask): return df.loc[mask].groupby('household_key')['sales_value'].sum()
    tot84 = t.groupby('household_key')['sales_value'].sum()
    tot28 = t[t['day'] > s-28].groupby('household_key')['sales_value'].sum()
    pm84 = S(tm, tm['f']==1); pm28 = S(tm, (tm['f']==1)&(tm['day']>s-28))
    pd84 = S(tm, tm['df']==1); pml84 = S(tm, tm['mf']==1)
    pc84 = S(tc, tc['cf']==1); pc28 = S(tc, (tc['cf']==1)&(tc['day']>s-28))
    wks = tm.loc[tm['f']==1].groupby('household_key')['week_no'].nunique()
    r = lambda x: x.reindex(hh).fillna(0.0)
    out['pm_spend84'] = r(pm84); out['pm_spend28'] = r(pm28)
    t84 = r(tot84).values; t28v = r(tot28).values
    out['pm_share84'] = np.where(t84>0, r(pm84).values/np.maximum(t84,1e-9), 0.0)
    out['pm_share28'] = np.where(t28v>0, r(pm28).values/np.maximum(t28v,1e-9), 0.0)
    out['pdisp_share84'] = np.where(t84>0, r(pd84).values/np.maximum(t84,1e-9), 0.0)
    out['pmail_share84'] = np.where(t84>0, r(pml84).values/np.maximum(t84,1e-9), 0.0)
    out['pc_share84'] = np.where(t84>0, r(pc84).values/np.maximum(t84,1e-9), 0.0)
    out['pc_share28'] = np.where(t28v>0, r(pc28).values/np.maximum(t28v,1e-9), 0.0)
    out['p_wks84'] = (r(wks)/12.0).clip(0,1)
    return out

feat = A.build_features(build)
print('built', feat.shape, 'time', round(time.time()-t0,1))
print(feat[NEW].describe().round(3).to_string())
# proxy ridge: base e008 vs +new
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float); sd = m['snapshot_day'].values
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
m = m.merge(feat, on=['household_key','snapshot_day'], how='left')
def run(cols, lam=100):
    X=m[cols].astype(float); X=X.fillna(X.median()); mu=X.mean(0); sg=X.std(0)+1e-9
    Xz=(X.values-mu.values)/sg.values
    tr=sd<=347; ev=sd>=375; Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
    w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
    return round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
print('base:', run(feats))
print('+promo:', run(feats+NEW))
path = A.save_table(feat, 'e015_promo')
print('saved', path)
