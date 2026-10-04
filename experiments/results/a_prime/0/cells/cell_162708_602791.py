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

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128), verbose=False):
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

# ---- block-history shape features from raw transactions (offline, capped at 459) ----
v = agent_api.snapshot()
t = v.transactions[['household_key','day','sales_value']]
first = t.groupby('household_key').day.min().rename('first_day')
def win_spend(lo,hi):
    return t[(t.day>=lo)&(t.day<=hi)].groupby('household_key').sales_value.sum()
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
rows=[]
for d in days:
    elig=first[first<=d-84].index
    blocks={}
    for k in range(1,13):
        blocks[k]=win_spend(d-28*k+1, d-28*(k-1)).reindex(elig).fillna(0.0)
    B=pd.DataFrame(blocks)  # cols 1..12, block1 = most recent
    f=pd.DataFrame(index=elig)
    f['b_med6']=B[[1,2,3,4,5,6]].median(axis=1)
    f['b_med12']=B.median(axis=1)
    f['b_mean6']=B[[1,2,3,4,5,6]].mean(axis=1)
    f['b_mean12']=B.mean(axis=1)
    f['b_pos_share12']=(B>0).mean(axis=1)
    f['b_pos_share6']=(B[[1,2,3,4,5,6]]>0).mean(axis=1)
    zs=0
    for k in range(1,13):
        zs=zs+(B[k]==0)*1*0  # placeholder
    # consecutive zero blocks ending at block1
    zst=np.zeros(len(elig))
    Bv=B.values
    for i in range(len(elig)):
        c=0
        for k in range(12):
            if Bv[i,k]==0: c+=1
            else: break
        zst[i]=c
    f['b_zerostreak']=zst
    f['b_max12']=B.max(axis=1)
    f['b_std12']=B.std(axis=1)
    f['b_med6_x_pos']=f['b_med6']*f['b_pos_share6']
    f['b_mean_pos']=B.replace(0,np.nan).mean(axis=1).fillna(0)
    f['b_med_div_mean']=f['b_med6']/f['b_mean6'].replace(0,np.nan)
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
shape=pd.concat(rows,ignore_index=True)
agent_api.save_table(shape,"shape_v1_offline.parquet")
print("shape",shape.shape)

df2 = tt.merge(base,on=keys,how="left").merge(shape,on=keys,how="left")
scols=[c for c in shape.columns if c not in keys+['index']]
m_ref=ridge_eval(df2,cols0,"ref")
ridge_eval(df2,cols0+scols,"e013+shape",verbose=True)
ridge_eval(df2,scols,"shape only")
# small: core + shape
core=[c for c in cols0 if c in ['x_exp4w','spend_28','spend_84','spend_364','baskets_28','recency','x_spend_p1','x_spend_p2','active_28','x_bv_4','x_r_4_8']]
ridge_eval(df2,core+scols,"core+shape")
ridge_eval(df2,core,"core")
