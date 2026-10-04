import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
m = A.load_saved('e013_stock.parquet')
snap_days=[431,459]; parts=[]
for sd in snap_days:
    v=A.snapshot(min(sd,459)); tr=v.transactions; tr=tr[tr.day<=sd].copy()
    hh_codes,hh_uniq=pd.factorize(tr.household_key)
    day=tr.day.values; sv=tr.sales_value.values
    n=len(hh_uniq)
    rec=pd.Series(sd-tr.groupby('household_key').day.max()).reindex(hh_uniq).fillna(999).values
    ten=pd.Series(sd-tr.groupby('household_key').day.min()).reindex(hh_uniq).values
    f=pd.DataFrame({'household_key':hh_uniq,'snapshot_day':sd})
    for hl in [7,14,28,56,112]:
        w=np.exp(np.log(0.5)*(sd-day)/hl)
        s=np.zeros(n); np.add.at(s,hh_codes,sv*w); f['ew%d'%hl]=s
    f['ew_ratio_7_56']=f.ew7/(f.ew56+0.01); f['ew_ratio_28_112']=f.ew28/(f.ew112+0.01)
    blocks=np.zeros((n,13))
    for k in range(1,14):
        msk=(day>sd-28*k)&(day<=sd-28*(k-1))
        b=np.zeros(n); np.add.at(b,hh_codes[msk],sv[msk]); blocks[:,k-1]=b
    bdf=pd.DataFrame(blocks,columns=['b%d'%k for k in range(1,14)])
    act=(bdf>0).mean(axis=1).values; nzmean=bdf.replace(0,np.nan).mean(axis=1).values
    nzmax=bdf.max(axis=1).values; nzmed=bdf.replace(0,np.nan).median(axis=1).values
    f['tp_act8']=act; f['tp_nzmean']=np.nan_to_num(nzmean); f['tp_nzmax']=nzmax
    f['tp_nzmed']=np.nan_to_num(nzmed); f['tp_pred']=act*np.nan_to_num(nzmean)
    f['tp_max_med']=nzmax/(np.nan_to_num(nzmed)+1); f['tp_last_max']=bdf.b1.values/(nzmax+0.01)
    f['tp_cv']=bdf.std(axis=1).values/(bdf.mean(axis=1).values+0.01)
    s28=bdf.b1.values+bdf.b2.values; s84=bdf.iloc[:,:3].values.sum(1); s364=bdf.values.sum(1)
    prof=np.column_stack([np.log1p(s28),np.log1p(s84),np.log1p(s364),rec,ten]).astype(float)
    mu=prof.mean(0); sdv=prof.std(0)+1e-9; Pz=(prof-mu)/sdv
    Pz=Pz/(np.linalg.norm(Pz,axis=1,keepdims=True)+1e-9); S=Pz@Pz.T; np.fill_diagonal(S,-9)
    idx=np.argpartition(-S,25,axis=1)[:,:25]; simsel=np.take_along_axis(S,idx,axis=1)
    ws=np.clip(simsel,0,None)**2; ws=ws/(ws.sum(1,keepdims=True)+1e-9)
    tgt=np.column_stack([s28,s84,s364,act,np.nan_to_num(nzmean)])
    peer=(ws[:,:,None]*tgt[idx]).sum(1)
    f['pr_s28']=peer[:,0]; f['pr_s84']=peer[:,1]; f['pr_s364']=peer[:,2]
    f['pr_act']=peer[:,3]; f['pr_nzmean']=peer[:,4]; f['pr_self_peer']=s28/(peer[:,0]+1)
    parts.append(f)
blk=pd.concat(parts,ignore_index=True)
newcols=[c for c in blk.columns if c not in ('household_key','snapshot_day')]
print('blk rows',len(blk),'cols',len(newcols))
df=m.merge(blk,on=['household_key','snapshot_day'],how='left')
print('notna ew28:',int(df.ew28.notna().sum()),'notna pr_s28:',int(df.pr_s28.notna().sum()))
path=A.save_table(df,'cand_v1.parquet'); print('saved',path)
# proxy eval
tt=A.train_targets(); d=df.merge(tt,on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
def prep(cols):
    X=d[cols].copy()
    for c in cols:
        if X[c].dtype==object or str(X[c].dtype)=='category': X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem):
    n=int(trm.sum()); Xd=np.hstack([np.ones((n,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@y[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
base_cols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
groups={'BASE':[],'EWMA':['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112'],
 'TWOPART':['tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv'],
 'PEER':['pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer'],'ALL':newcols}
for name,grp in groups.items():
    cols=base_cols+grp
    p=fitpred(prep(cols),300,tr431,m431)
    print(name,len(cols),'MAE@431',round(float(np.abs(p-y[m431]).mean()),3))
