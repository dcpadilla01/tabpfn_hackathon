
import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')

E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']

def make_features(view, snapshot_day):
    end = snapshot_day
    t = view.table('transactions'); t = t[t.day<=end]
    hh_all = view.households
    out = pd.DataFrame(index=hh_all)
    def wsum(lo, hi=None):
        hi = end if hi is None else hi
        return t[(t.day>end-lo)&(t.day<=hi)].groupby('household_key')['sales_value'].sum()
    for name, lo in [('spend_7',7),('spend_14',14),('spend_28',28),('spend_56',56),('spend_84',84),('spend_180',180),('spend_365',365)]:
        out[name] = wsum(lo)
    out['spend_28_prior'] = wsum(56, end-28)
    out['spend_84_prior'] = wsum(168, end-84)
    d = t.groupby(['household_key','day'])['sales_value'].sum().reset_index()
    d['age'] = end - d['day']
    for hl in [7,14,28,56,84,180]:
        wgt = 0.5**(d['age']/hl)
        out['ew_%d'%hl] = (d['sales_value']*wgt).groupby(d['household_key']).sum()
    for lag in [336,364,392]:
        out['spend_lag%d'%lag] = t[(t.day>end-lag-28)&(t.day<=end-lag)].groupby('household_key')['sales_value'].sum()
    total = t.groupby('household_key')['sales_value'].sum()
    first_day = t.groupby('household_key')['day'].min()
    tenure_wk = ((end-first_day)/7.0).replace(0,np.nan)
    out['longrun_wk'] = total/tenure_wk
    out['ratio28_lr'] = out['spend_28']/(out['longrun_wk']*4).replace(0,np.nan)
    out['ratio84_lr'] = out['spend_84']/(out['longrun_wk']*12).replace(0,np.nan)
    b84 = t[(t.day>end-84)&(t.day<=end)]
    bsum = b84.groupby(['household_key','basket_id'])['sales_value'].sum()
    bcnt = bsum.groupby('household_key').agg(baskets_84='size', basket_max_84='max', basket_std_84='std', basket_med_84='median')
    out = out.join(bcnt)
    for name, lo in [('baskets_28',28),('baskets_14',14),('baskets_7',7)]:
        out[name] = t[(t.day>end-lo)&(t.day<=end)].groupby('household_key')['basket_id'].nunique()
    out['days_since_last'] = end - t.groupby('household_key')['day'].max()
    out['days_since_first'] = end - first_day
    out['avg_basket_84'] = out['spend_84']/out['baskets_84'].replace(0,np.nan)
    out['trips_per_wk_84'] = out['baskets_84']/12.0
    out['spend_28_ratio'] = out['spend_28']/out['spend_28_prior'].replace(0,np.nan)
    out['n_products_84'] = b84.groupby('household_key')['product_id'].nunique()
    out['n_stores_84'] = b84.groupby('household_key')['store_id'].nunique()
    out['spend_trend'] = (out['spend_28']-out['spend_28_prior'])/28.0
    out['active_28'] = (out['baskets_28']>0).astype(float)
    out['active_days_28'] = t[(t.day>end-28)&(t.day<=end)].groupby('household_key')['day'].nunique()
    td = t[(t.day>end-180)&(t.day<=end)].groupby('household_key')['day'].unique()
    gaps = td.apply(lambda a: np.diff(np.sort(a)) if len(a)>1 else np.array([]))
    out['gap_mean_180'] = gaps.apply(lambda a: a.mean() if len(a)>0 else np.nan)
    out['gap_cv'] = gaps.apply(lambda a: (a.std()/a.mean()) if len(a)>1 and a.mean()>0 else np.nan)
    out['max_gap_180'] = gaps.apply(lambda a: a.max() if len(a)>0 else np.nan)
    out['recency_ratio'] = out['days_since_last']/out['gap_mean_180']
    blk = (end - t['day'])//7
    wk = t[blk<26].groupby(['household_key', blk[blk<26]])['sales_value'].sum()
    out['zero_weeks_26'] = 1 - wk.groupby('household_key').size()/26.0
    b28s = t[(t.day>end-28)&(t.day<=end)].groupby(['household_key','basket_id'])['sales_value'].sum()
    out['avg_basket_28'] = b28s.groupby('household_key').mean()
    out['basket_28_vs_84'] = out['avg_basket_28']/out['avg_basket_84']
    lines = b84.groupby('household_key').size()
    qty = b84.groupby('household_key')['quantity'].sum()
    out['lines_per_basket_84'] = lines/out['baskets_84'].replace(0,np.nan)
    out['units_per_basket_84'] = qty/out['baskets_84'].replace(0,np.nan)
    out['unit_price_84'] = out['spend_84']/qty.replace(0,np.nan)
    disc = b84.groupby('household_key')[['coupon_disc','retail_disc','coupon_match_disc']].sum().sum(axis=1)
    out['disc_share_84'] = -disc/out['spend_84'].replace(0,np.nan)
    out['zero_line_share_84'] = (b84['sales_value']==0).groupby(b84['household_key']).mean()
    prod = view.table('products')[['product_id','brand','department']]
    b84b = b84.merge(prod, on='product_id', how='left')
    pb = b84b[b84b.brand=='Private'].groupby('household_key')['sales_value'].sum()
    out['private_share_84'] = pb/out['spend_84'].replace(0,np.nan)
    for dep in ['GROCERY','DRUG GM','PRODUCE','MEAT','DELI','PASTRY','COSMETICS']:
        ds = b84b[b84b.department==dep].groupby('household_key')['sales_value'].sum()
        out['dep_%s'%dep[:6].replace(' ','')] = ds/out['spend_84'].replace(0,np.nan)
    p84s = b84.groupby(['household_key','product_id'])['sales_value'].sum()
    prev = t[(t.day>end-365)&(t.day<=end-84)].groupby(['household_key','product_id']).size().rename('p')
    j = p84s.to_frame('s').join(prev, how='left')
    rep = j[j.p.notna()].groupby('household_key')['s'].sum()
    out['repeat_share_84'] = rep/out['spend_84'].replace(0,np.nan)
    ss = b84.groupby(['household_key','store_id'])['sales_value'].sum()
    out['top_store_share_84'] = ss.groupby('household_key').max()/out['spend_84'].replace(0,np.nan)
    out['sin_y'] = np.sin(2*np.pi*end/364); out['cos_y'] = np.cos(2*np.pi*end/364)
    out['sin_q'] = np.sin(2*np.pi*end/91); out['cos_q'] = np.cos(2*np.pi*end/91)
    out['day_idx'] = end/100.0
    out['ix_ew28_trips'] = out['ew_28']*out['trips_per_wk_84']
    out['ix_spend84_active'] = out['spend_84']*out['active_28']
    out['ix_recency_ew'] = out['ew_28']*np.exp(-out['days_since_last']/28.0)
    return out.reindex(hh_all)

t0=time.time()
F = agent_api.build_features(make_features)
print('built in %.0fs' % (time.time()-t0), F.shape)
missing = [c for c in E5COLS if c not in F.columns]
print('missing E5 cols:', missing)
agent_api.save_table(F, 'e008_candidate')

# proxy CV
tt = agent_api.train_targets()
df = F.merge(tt, on=['household_key','snapshot_day'])
itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
y = df['future_spend_4w'].values
def ridge(Xtr,ytr,Xva,lam,clip=8.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1
    A=np.clip((Xtr-mu)/sd,-clip,clip); B=np.clip((Xva-mu)/sd,-clip,clip)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    return B@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@ytr)
def ev(cols):
    X = df[cols].replace([np.inf,-np.inf],np.nan)
    X = X.fillna(X[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr],y[itr],X[iva],100.0)
    return np.abs(p-y[iva]).mean()

groups = {
 'A_recency': ['baskets_7','baskets_14','active_days_14','recency_ratio','max_gap_180','gap_mean_180'],
 'B_intermittent': ['zero_weeks_26'],
 'C_basket': ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84'],
 'D_disc_brand': ['disc_share_84','zero_line_share_84','private_share_84'],
 'E_mix': ['dep_GROCER','dep_DRUGG','dep_PRODUC','dep_MEAT','dei_DELI','dep_PASTRY','dep_COSMET'],
 'F_loyalty': ['repeat_share_84','top_store_share_84'],
 'G_season': ['sin_y','cos_y','sin_q','cos_q','day_idx'],
 'H_interact': ['ix_ew28_trips','ix_spend84_active','ix_recency_ew'],
}
base = [c for c in E5COLS if c in F.columns]
print('rebuild-E005 proxy MAE %.3f (n=%d)' % (ev(base), len(base)))
allnew = []
for g,cols in groups.items():
    cols = [c for c in cols if c in F.columns]
    allnew += cols
    print('%-14s +%-3d -> %.3f' % (g, len(cols), ev(base+cols)))
print('ALL new (%d) -> %.3f' % (len(allnew), ev(base+allnew)))
