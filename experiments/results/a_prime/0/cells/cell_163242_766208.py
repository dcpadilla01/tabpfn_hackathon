import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)

def ridge_eval(frame, feat_cols, tag, alphas=(32,64,128,256,512), verbose=False):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    Xf=fit[feat_cols].astype(float).values; Xh=hold[feat_cols].astype(float).values
    med=np.nanmedian(Xf,axis=0); Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9; Xf=(Xf-mu)/sd; Xh=(Xh-mu)/sd
    Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
    itr=(fit.snapshot_day<=375).values; iva=(fit.snapshot_day==403).values
    best=None
    for a in alphas:
        A=Xf[itr]; w=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@yf[itr])
        mae=np.abs(np.clip(Xf[iva]@w,0,None)-yf[iva]).mean()
        if best is None or mae<best[1]: best=(a,mae)
    a=best[0]; G=Xf.T@Xf+a*np.eye(Xf.shape[1]); w=np.linalg.solve(G,Xf.T@yf)
    ph=np.clip(Xh@w,0,None); mae=np.abs(ph-yh).mean()
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

v=agent_api.snapshot()
t=v.transactions[['household_key','day','sales_value','product_id']].merge(
    v.products[['product_id','department']].astype({'department':'str'}),on='product_id',how='left')
t['department']=t['department'].fillna('other').astype(str)
CONSUM={'GROCERY','MEAT','PRODUCE','DELI','MEAT-PCKGD','SEAFOOD','SEAFOOD-PCKGD','PASTRY','DAIRY','FROZEN','BAKERY','KIOSK-GAS'}
t['is_cons']=t.department.isin(CONSUM).astype(float)
t['dept8']=np.where(t.department.isin(['GROCERY','DRUG GM','MEAT','PRODUCE','MEAT-PCKGD','DELI','KIOSK-GAS','PASTRY']),t.department,'other')
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
first=t.groupby('household_key').day.min()
rows=[]
for d in days:
    elig=first[first<=d-84].index
    w28=t[(t.day>=d-27)&(t.day<=d)]; w84=t[(t.day>=d-83)&(t.day<=d)]
    f=pd.DataFrame(index=elig)
    s28=w28.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0)
    s84=w84.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0)
    s28=s28.reindex(elig).fillna(0.0); s84=s84.reindex(elig).fillna(0.0)
    for c in s28.columns:
        f[f'd28_{c}']=s28[c].values
        f[f'r2884_{c}']=(s28[c]/s84[c].replace(0,np.nan)).fillna(1.0).values
    c28=w28.groupby('household_key').is_cons.sum().reindex(elig).fillna(0.0)
    a28=w28.groupby('household_key').sales_value.sum().reindex(elig).fillna(0.0)
    f['cons_share28']=(c28/a28.replace(0,np.nan)).fillna(0).values
    c84=w84.groupby('household_key').is_cons.sum().reindex(elig).fillna(0.0)
    a84=w84.groupby('household_key').sales_value.sum().reindex(elig).fillna(0.0)
    f['cons_share84']=(c84/a84.replace(0,np.nan)).fillna(0).values
    lastc=w28[w28.is_cons==1].groupby('household_key').day.max()
    f['days_since_cons']=(d-lastc).reindex(elig).fillna(28).values
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
dept=pd.concat(rows,ignore_index=True)
agent_api.save_table(dept,"dept_v1_offline.parquet")
print("dept feats:",dept.shape, list(dept.columns)[:8])
dfD=tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left")
dcols=[c for c in dept.columns if c not in keys+['index']]
ridge_eval(dfD,cols0+dcols,"e013+dept",verbose=True)
