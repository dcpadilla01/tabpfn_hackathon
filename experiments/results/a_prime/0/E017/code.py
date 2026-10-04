import agent_api as A, pandas as pd, numpy as np
for name in ['e013_stock','rhythm_v1','stock_v1','mkt_v2','e011_rank']:
    t = A.load_saved(name+'.parquet')
    print('==', name, t.shape)
    print(list(t.columns))
tt = A.train_targets()
print(tt['future_spend_4w'].describe())
print('zero share', float((tt['future_spend_4w']==0).mean()))
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

t = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].copy()
for c in feat_cols:
    if X[c].dtype == object or str(X[c].dtype)=='category':
        X[c] = X[c].astype('category').cat.codes.astype(float)
Xv = X.values.astype(float)
med = np.nanmedian(Xv, axis=0)
Xv = np.where(np.isnan(Xv), med, Xv)
mu, sd = Xv.mean(0), Xv.std(0)+1e-9
Xs = (Xv-mu)/sd

def fit(Xs, y, lam):
    n = len(Xs); Xd = np.hstack([np.ones((n,1)), Xs])
    M = Xd.T@Xd + lam*np.eye(Xd.shape[1]); M[0,0] -= lam
    return np.linalg.solve(M, Xd.T@y)
def pred(w, Xs):
    return np.hstack([np.ones((len(Xs),1)), Xs])@w

fit_mask = df.snapshot_day<=375; int_mask = df.snapshot_day==403
best = None
for lam in [1,3,10,30,100,300]:
    w = fit(Xs[fit_mask], y[fit_mask], lam)
    mae = np.abs(pred(w, Xs[int_mask]) - y[int_mask]).mean()
    if best is None or mae < best[1]: best = (lam, round(mae,3))
print('internal lambda/MAE@403:', best)
lam = best[0]
tr = df.snapshot_day<=403
w = fit(Xs[tr], y[tr], lam)
m431 = df.snapshot_day==431
p = pred(w, Xs[m431]); yy = y[m431]
print('pseudo-val MAE@431:', round(np.abs(p-yy).mean(),3), 'n=', m431.sum())
print('baseline spend_28:', round(np.abs(df.loc[m431,'spend_28'].values-yy).mean(),3),
      '| x_exp4w:', round(np.abs(df.loc[m431,'x_exp4w'].values-yy).mean(),3),
      '| zero:', round(np.abs(yy).mean(),3))
r = pd.DataFrame({'y':yy,'p':p}); r['ae']=abs(r.p-r.y); r['bias']=r.p-r.y
r['bin'] = pd.qcut(r.y, 10, duplicates='drop')
print(r.groupby('bin', observed=True).agg(n=('y','size'), ymean=('y','mean'), pmean=('p','mean'), mae=('ae','mean'), bias=('bias','mean')).round(1))
z = r.y==0
print('zero-true rows:', z.sum(), 'mean pred on them:', round(r.loc[z,'p'].mean(),2), 'MAE there:', round(r.loc[z,'ae'].mean(),2))
nz = ~z
print('nonzero rows MAE:', round(r.loc[nz,'ae'].mean(),2))
top = r.nlargest(15,'ae')
cols = ['spend_28','spend_84','spend_364','recency','active_28','rs_active_share_84','stk_w1_share','x_exp4w']
print(pd.concat([top, df.loc[m431, cols].loc[top.index]], axis=1).round(1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

snap_days = [431, 459]
base = A.load_saved('e013_stock.parquet')
out = {}
for sd in snap_days:
    v = A.snapshot(min(sd,459))
    tr = v.transactions
    tr = tr[tr.day <= sd]
    hh = tr.household_key.values
    # per-household arrays
    g = tr.groupby('household_key')
    spend_all = g.sales_value.sum()
    day = tr.day.values; sv = tr.sales_value.values
    hh_codes, hh_uniq = pd.factorize(tr.household_key)
    # EWMA half-lives
    ew = {}
    for hl in [7,14,28,56,112]:
        w = np.exp(np.log(0.5)*(sd-day)/hl)
        s = np.zeros(len(hh_uniq)); np.add.at(s, hh_codes, sv*w)
        ew[hl] = pd.Series(s, index=hh_uniq)
    # aligned prior 4w blocks (t-28k+1..t-28k), k=1..13
    blocks = np.zeros((len(hh_uniq), 13))
    for k in range(1,14):
        m = (day > sd-28*k) & (day <= sd-28*(k-1))
        b = np.zeros(len(hh_uniq)); np.add.at(b, hh_codes[m], sv[m])
        blocks[:,k-1] = b
    bdf = pd.DataFrame(blocks, index=hh_uniq, columns=[f'b{k}' for k in range(1,14)])
    # two-part: active share over prior 8 blocks; conditional mean of nonzero blocks
    act = (bdf>0).mean(axis=1)
    nzmean = bdf.replace(0, np.nan).mean(axis=1)
    nzmax = bdf.max(axis=1); nzmed = bdf.replace(0,np.nan).median(axis=1)
    f = pd.DataFrame(index=hh_uniq)
    f['ew7']=ew[7]; f['ew14']=ew[14]; f['ew28']=ew[28]; f['ew56']=ew[56]; f['ew112']=ew[112]
    f['ew_ratio_7_56']=ew[7]/(ew[56]+0.01); f['ew_ratio_28_112']=ew[28]/(ew[112]+0.01)
    f['tp_act8']=act; f['tp_nzmean']=nzmean.fillna(0); f['tp_nzmax']=nzmax; f['tp_nzmed']=nzmed.fillna(0)
    f['tp_pred']=act*nzmean.fillna(0)
    f['tp_max_med']=nzmax/(nzmed.fillna(0)+1)
    f['tp_last_max']=bdf.b1/(nzmax+0.01)
    f['tp_cv']=bdf.std(axis=1)/(bdf.mean(axis=1)+0.01)
    # peer similarity
    prof = pd.DataFrame(index=hh_uniq)
    s28 = bdf.b1+bdf.b2; s84 = bdf.iloc[:,:3].sum(axis=1); s364=bdf.sum(axis=1)
    rec = sd - g.day.max()
    prof['l_s28']=np.log1p(s28); prof['l_s84']=np.log1p(s84); prof['l_s364']=np.log1p(s364)
    prof['rec']=rec.reindex(hh_uniq).fillna(999).values
    ten = sd - g.day.min()
    prof['ten']=ten.reindex(hh_uniq).values
    Pv = prof.values.astype(float)
    Pv = np.nan_to_num(Pv, nan=0.0)
    mu=Pv.mean(0); sdv=Pv.std(0)+1e-9; Pz=(Pv-mu)/sdv
    Pz = Pz/ (np.linalg.norm(Pz,axis=1,keepdims=True)+1e-9)
    S = Pz@Pz.T
    np.fill_diagonal(S, -9)
    k=25
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    simsel = np.take_along_axis(S, idx, axis=1)
    ws = np.clip(simsel,0,None)**2
    ws = ws/(ws.sum(1,keepdims=True)+1e-9)
    tgt = np.column_stack([s28.values, s84.values, s364.values, act.values, nzmean.fillna(0).values])
    peer = (ws[:,:,None]*tgt[idx]).sum(1)
    f['pr_s28']=peer[:,0]; f['pr_s84']=peer[:,1]; f['pr_s364']=peer[:,2]
    f['pr_act']=peer[:,3]; f['pr_nzmean']=peer[:,4]
    f['pr_self_peer']=s28.values/(peer[:,0]+1)
    out[sd]=f
    print(sd, f.shape, 'hh with data:', len(hh_uniq))

blocks = []
for sd in snap_days:
    b = out[sd].reset_index().rename(columns={'index':'household_key'})
    b.insert(1,'snapshot_day',sd)
    blocks.append(b)
blk = pd.concat(blocks)
cols = [c for c in blk.columns if c not in ('household_key','snapshot_day')]
m = base.merge(blk, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape)

# ridge proxy eval
tt = A.train_targets()
df = m.merge(tt, on=['household_key','snapshot_day'])
y = df.future_spend_4w.values
def prep(cols):
    X = df[cols].copy()
    for c in cols:
        if X[c].dtype==object or str(X[c].dtype)=='category': X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv = X.values.astype(float)
    med = np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    mu,s = Xv.mean(0), Xv.std(0)+1e-9
    return (Xv-mu)/s
def fitpred(Xs,y,lam,tr,te):
    n=len(tr); Xd=np.hstack([np.ones((n,1)),Xs[tr]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@y[tr])
    return np.hstack([np.ones((len(te),1)),Xs[te]])@w
base_cols=[c for c in base.columns if c not in ('household_key','snapshot_day')]
m431=df.snapshot_day==431; tr431=df.snapshot_day<=403
for lam in [30,100,300]:
    p=fitpred(prep(base_cols),y,lam,tr431,m431)
    print('lam',lam,'base MAE@431',round(np.abs(p-y[m431]).mean(),3))
for grp,name in [(cols,'ALLnew'),(['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112'],'EWMA'),
                 (['tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv'],'TWOPART'),
                 (['pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer'],'PEER')]:
    p=fitpred(prep(base_cols+grp),y,300,tr431,m431)
    print(name,'MAE@431',round(np.abs(p-y[m431]).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

m = A.load_saved('e013_stock.parquet')
# rebuild candidate block quickly from saved merge? It wasn't saved; recompute minimal (reuse code)
snap_days=[431,459]; out={}
for sd in snap_days:
    v=A.snapshot(min(sd,459)); tr=v.transactions; tr=tr[tr.day<=sd]
    hh_codes,hh_uniq=pd.factorize(tr.household_key)
    day=tr.day.values; sv=tr.sales_value.values; g=tr.groupby('household_key')
    f=pd.DataFrame(index=hh_uniq)
    for hl in [7,14,28,56,112]:
        w=np.exp(np.log(0.5)*(sd-day)/hl)
        s=np.zeros(len(hh_uniq)); np.add.at(s,hh_codes,sv*w); f['ew%d'%hl]=s
    f['ew_ratio_7_56']=f.ew7/(f.ew56+0.01); f['ew_ratio_28_112']=f.ew28/(f.ew112+0.01)
    blocks=np.zeros((len(hh_uniq),13))
    for k in range(1,14):
        msk=(day>sd-28*k)&(day<=sd-28*(k-1))
        b=np.zeros(len(hh_uniq)); np.add.at(b,hh_codes[msk],sv[msk]); blocks[:,k-1]=b
    bdf=pd.DataFrame(blocks,index=hh_uniq,columns=['b%d'%k for k in range(1,14)])
    act=(bdf>0).mean(axis=1); nzmean=bdf.replace(0,np.nan).mean(axis=1)
    nzmax=bdf.max(axis=1); nzmed=bdf.replace(0,np.nan).median(axis=1)
    f['tp_act8']=act; f['tp_nzmean']=nzmean.fillna(0); f['tp_nzmax']=nzmax; f['tp_nzmed']=nzmed.fillna(0)
    f['tp_pred']=act*nzmean.fillna(0); f['tp_max_med']=nzmax/(nzmed.fillna(0)+1)
    f['tp_last_max']=bdf.b1/(nzmax+0.01); f['tp_cv']=bdf.std(axis=1)/(bdf.mean(axis=1)+0.01)
    s28=bdf.b1+bdf.b2; s84=bdf.iloc[:,:3].sum(axis=1); s364=bdf.sum(axis=1)
    rec=(sd-g.day.max()).reindex(hh_uniq).fillna(999).values; ten=(sd-g.day.min()).reindex(hh_uniq).values
    prof=pd.DataFrame({'l_s28':np.log1p(s28),'l_s84':np.log1p(s84),'l_s364':np.log1p(s364),'rec':rec,'ten':ten},index=hh_uniq)
    Pv=prof.values.astype(float); mu=Pv.mean(0); sdv=Pv.std(0)+1e-9; Pz=(Pv-mu)/sdv
    Pz=Pz/(np.linalg.norm(Pz,axis=1,keepdims=True)+1e-9); S=Pz@Pz.T; np.fill_diagonal(S,-9)
    idx=np.argpartition(-S,25,axis=1)[:,:25]; simsel=np.take_along_axis(S,idx,axis=1)
    ws=np.clip(simsel,0,None)**2; ws=ws/(ws.sum(1,keepdims=True)+1e-9)
    tgt=np.column_stack([s28.values,s84.values,s364.values,act.values,nzmean.fillna(0).values])
    peer=(ws[:,:,None]*tgt[idx]).sum(1)
    f['pr_s28']=peer[:,0]; f['pr_s84']=peer[:,1]; f['pr_s364']=peer[:,2]; f['pr_act']=peer[:,3]
    f['pr_nzmean']=peer[:,4]; f['pr_self_peer']=s28.values/(peer[:,0]+1)
    out[sd]=f
blk=pd.concat([out[sd].reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=sd) for sd in snap_days])
newcols=[c for c in blk.columns if c not in ('household_key','snapshot_day')]
df=m.merge(blk,on=['household_key','snapshot_day'],how='left')
tt=A.train_targets(); df=df.merge(tt,on=['household_key','snapshot_day'])
y=df.future_spend_4w.values
def prep(cols):
    X=df[cols].copy()
    for c in cols:
        if X[c].dtype==object or str(X[c].dtype)=='category': X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,y,lam,trm,tem):
    n=int(trm.sum()); Xd=np.hstack([np.ones((n,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@y[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
base_cols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
m431=(df.snapshot_day==431).values; tr431=(df.snapshot_day<=403).values
for name,grp in [('BASE',[]),('EWMA',['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']),
    ('TWOPART',['tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv']),
    ('PEER',['pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer']),
    ('ALLNEW',newcols)]:
    p=fitpred(prep(base_cols+grp),y,300,tr431,m431)
    print(name,'MAE@431',round(np.abs(p-y[m431]).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
m = A.load_saved('e013_stock.parquet')
print(m.dtypes.value_counts())
print('m household dtype', m.household_key.dtype, m.household_key.head(3).tolist())
v=A.snapshot(459); print('tx household dtype', v.transactions.household_key.dtype, v.transactions.household_key.head(3).tolist())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
m = A.load_saved('e013_stock.parquet')
# quick rebuild of just EWMA at 431 to test merge
sd=431
v=A.snapshot(459); tr=v.transactions; tr=tr[tr.day<=sd]
hh_codes,hh_uniq=pd.factorize(tr.household_key)
day=tr.day.values; sv=tr.sales_value.values
w=np.exp(np.log(0.5)*(sd-day)/28)
s=np.zeros(len(hh_uniq)); np.add.at(s,hh_codes,sv*w)
f=pd.DataFrame({'household_key':hh_uniq,'snapshot_day':sd,'ew28x':s})
print(f.head(3))
mm=m[m.snapshot_day==sd].merge(f,on=['household_key','snapshot_day'],how='left')
print('non-null ew28x:', mm.ew28x.notna().sum(), 'of', len(mm))
print(mm[['household_key','spend_28','ew28x']].head())
print('corr', mm[['spend_28','ew28x']].corr().iloc[0,1])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
m = A.load_saved('e013_stock.parquet')
snap_days=[431,459]; out={}
for sd in snap_days:
    v=A.snapshot(min(sd,459)); tr=v.transactions; tr=tr[tr.day<=sd]
    hh_codes,hh_uniq=pd.factorize(tr.household_key)
    day=tr.day.values; sv=tr.sales_value.values
    f=pd.DataFrame(index=hh_uniq)
    w=np.exp(np.log(0.5)*(sd-day)/28)
    s=np.zeros(len(hh_uniq)); np.add.at(s,hh_codes,sv*w); f['ew28']=s
    out[sd]=f
    print(sd,'index name:',f.index.name)
blk=pd.concat([out[sd].reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=sd) for sd in snap_days])
print(blk.columns.tolist(), blk.head(3))
df=m.merge(blk,on=['household_key','snapshot_day'],how='left')
print('ew28 notna:', df.ew28.notna().sum(), 'rows', len(df))


# ---- cell ----
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


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v1.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
y=d.future_spend_4w.values
print(d[['ew28','pr_s28','tp_act8','tp_nzmean','y' if 'y' in d else 'future_spend_4w']].describe().round(2))
sub=d[m431]
print('corr with y@431:')
for c in ['spend_28','ew28','tp_pred','pr_s28','tp_nzmean','tp_act8']:
    print(c, round(float(np.corrcoef(sub[c].fillna(0),sub.future_spend_4w)[0,1]),3))
def prep1(cols):
    X=d[cols].copy()
    for c in cols:
        if X[c].dtype==object: X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam):
    n=int(tr431.sum()); Xd=np.hstack([np.ones((n,1)),Xs[tr431]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@y[tr431])
    return np.hstack([np.ones((int(m431.sum()),1)),Xs[m431]])@w
for c in ['spend_28','ew28','tp_pred','pr_s28']:
    p=fitpred(prep1([c]),10)
    print('single',c,'MAE@431',round(float(np.abs(p-y[m431]).mean()),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
m = A.load_saved('e013_stock.parquet')
SNAPS = [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]
parts=[]
for sd in SNAPS:
    v=A.snapshot(min(sd,459)); tr=v.transactions
    tr=tr[tr.day<=sd]
    hh_codes,hh_uniq=pd.factorize(tr.household_key)
    day=tr.day.values; sv=tr.sales_value.values; n=len(hh_uniq)
    g=tr.groupby('household_key')
    rec=pd.Series(sd-g.day.max()).reindex(hh_uniq).fillna(999).values
    ten=pd.Series(sd-g.day.min()).reindex(hh_uniq).values
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
    act=(bdf>0).mean(axis=1).values
    nzmean=bdf.replace(0,np.nan).mean(axis=1).values
    nzmed=bdf.replace(0,np.nan).median(axis=1).values
    nzmax=bdf.max(axis=1).values
    f['tp_act8']=act; f['tp_nzmean']=np.nan_to_num(nzmean); f['tp_nzmax']=nzmax
    f['tp_nzmed']=np.nan_to_num(nzmed); f['tp_pred']=act*np.nan_to_num(nzmean)
    f['tp_max_med']=nzmax/(np.nan_to_num(nzmed)+1); f['tp_last_max']=bdf.b1.values/(nzmax+0.01)
    f['tp_cv']=bdf.std(axis=1).values/(bdf.mean(axis=1).values+0.01)
    f['own_ratio_med']=bdf.b1.values/(np.nan_to_num(nzmed)+1)
    ownp=np.zeros(n)
    Bv=bdf.values
    for i in range(n):
        row=Bv[i]; pos=row[row>0]
        ownp[i]=(pos<=row[0]).mean() if row[0]>0 and len(pos)>0 else 0.0
    f['own_pct']=ownp
    s28=bdf.b1.values+bdf.b2.values; s84=bdf.iloc[:,:3].values.sum(1); s364=bdf.values.sum(1)
    prof=np.column_stack([np.log1p(s28),np.log1p(s84),np.log1p(s364),rec,ten]).astype(float)
    mu=prof.mean(0); sdv=prof.std(0)+1e-9; Pz=(prof-mu)/sdv
    Pz=Pz/(np.linalg.norm(Pz,axis=1,keepdims=True)+1e-9); S=Pz@Pz.T; np.fill_diagonal(S,-9)
    k=min(25,n-1)
    idx=np.argpartition(-S,k,axis=1)[:,:k]; simsel=np.take_along_axis(S,idx,axis=1)
    ws=np.clip(simsel,0,None)**2; ws=ws/(ws.sum(1,keepdims=True)+1e-9)
    tgt=np.column_stack([s28,s84,s364,act,np.nan_to_num(nzmean)])
    peer=(ws[:,:,None]*tgt[idx]).sum(1)
    f['pr_s28']=peer[:,0]; f['pr_s84']=peer[:,1]; f['pr_s364']=peer[:,2]
    f['pr_act']=peer[:,3]; f['pr_nzmean']=peer[:,4]; f['pr_self_peer']=s28/(peer[:,0]+1)
    parts.append(f)
blk=pd.concat(parts,ignore_index=True)
newcols=[c for c in blk.columns if c not in ('household_key','snapshot_day')]
df=m.merge(blk,on=['household_key','snapshot_day'],how='left')
print('merged',df.shape,'notna ew28:',int(df.ew28.notna().sum()))
path=A.save_table(df,'cand_v2.parquet'); print('saved',path)
tt=A.train_targets(); d=df.merge(tt,on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
def prep(cols):
    X=d[cols].copy()
    for c in cols:
        if X[c].dtype==object or str(X[c].dtype)=='category': X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem):
    nn=int(trm.sum()); Xd=np.hstack([np.ones((nn,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@y[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
base_cols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
groups={'BASE':[],'EWMA':['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112'],
 'TWOPART':['tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv','own_pct','own_ratio_med'],
 'PEER':['pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer'],'ALL':newcols}
for name,grp in groups.items():
    p=fitpred(prep(base_cols+grp),300,tr431,m431)
    print(name,len(base_cols+grp),'MAE@431',round(float(np.abs(p-y[m431]).mean()),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v2.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
base_cols=[c for c in A.load_saved('e013_stock.parquet').columns if c not in ('household_key','snapshot_day')]
newcols=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112','tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv','own_ratio_med','own_pct','pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer']
allc=base_cols+newcols
def prep(cols,frame):
    X=frame[cols].copy()
    for c in cols:
        if X[c].dtype==object or str(X[c].dtype)=='category': X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem,yy):
    nn=int(trm.sum()); Xd=np.hstack([np.ones((nn,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@yy[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
Xs=prep(allc,d)
p=fitpred(Xs,300,tr431,m431,y)
r=y[m431]-p
res=pd.Series(r)
sub=d[m431]
cors={}
for c in allc:
    v=sub[c].astype(float).fillna(sub[c].astype(float).median() if sub[c].dtype!=object else 0).values
    if sub[c].dtype==object: v=sub[c].astype('category').cat.codes.values.astype(float)
    s=np.std(v)
    if s>1e-9: cors[c]=float(np.corrcoef(v,r)[0,1])
cs=pd.Series(cors).sort_values()
print('TOP NEG residual-corr:'); print(cs.head(12).round(3))
print('TOP POS residual-corr:'); print(cs.tail(15).round(3))
# tail block built from cohort percentiles at 431
q90_28=sub.spend_28.quantile(.9); q90_84=sub.spend_84.quantile(.9); q90_364=sub.spend_364.quantile(.9)
tb=pd.DataFrame(index=sub.index)
tb['t_s28_q90']=sub.spend_28/q90_28; tb['t_s84_q90']=sub.spend_84/q90_84; tb['t_s364_q90']=sub.spend_364/q90_364
tb['t_top5_s28']=(sub.spend_28>=sub.spend_28.quantile(.95)).astype(float)
tb['t_top5_s364']=(sub.spend_364>=sub.spend_364.quantile(.95)).astype(float)
tb['t_sq28']=np.sqrt(sub.spend_28); tb['t_log28']=np.log1p(sub.spend_28)
tb['t_rk_x_q90']=sub.rk_s28_pct*tb.t_s28_q90
# residual corr of tail feats
tc={}
for c in tb.columns: tc[c]=float(np.corrcoef(tb[c].fillna(0).values,r)[0,1])
print('tail-feat residual corr:', pd.Series(tc).round(3).to_dict())
# test: base+EWMA, base+tail, base+EWMA+tail (tail built per-snapshot for all snaps quickly via groupby transform)
g28=d.groupby('snapshot_day').spend_28.transform(lambda s: s.quantile(.9))
g84=d.groupby('snapshot_day').spend_84.transform(lambda s: s.quantile(.9))
g364=d.groupby('snapshot_day').spend_364.transform(lambda s: s.quantile(.95))
d['t_s28_q90']=d.spend_28/g28; d['t_s84_q90']=d.spend_84/g84; d['t_s364_q95']=d.spend_364/g364
d['t_top5_s28']=(d.spend_28>=d.groupby('snapshot_day').spend_28.transform(lambda s:s.quantile(.95))).astype(float)
d['t_sq28']=np.sqrt(d.spend_28)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_sq28']
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
for name,grp in [('BASE',[]),('+EWMA',ew),('+TAIL',tail),('+EWMA+TAIL',ew+tail),('+TAIL+rkx',['t_rk_x_q90'] if False else tail+['t_rk_x_q90'])]:
    X2=prep(allc if False else base_cols+newcols+ [c for c in grp if c in d.columns],d) if False else prep(base_cols+grp,d)
    p2=fitpred(X2,300,tr431,m431,y)
    print(name,'MAE@431',round(float(np.abs(p2-y[m431]).mean()),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v2.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
base_cols=[c for c in A.load_saved('e013_stock.parquet').columns if c not in ('household_key','snapshot_day')]
newcols=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112','tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv','own_ratio_med','own_pct','pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer']
allc=base_cols+newcols
def prep(cols,frame):
    X=frame[cols].copy()
    for c in cols:
        if X[c].dtype==object: X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem,yy):
    nn=int(trm.sum()); Xd=np.hstack([np.ones((nn,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@yy[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
Xs=prep(allc,d)
p=fitpred(Xs,300,tr431,m431,y)
r=y[m431]-p
sub=d[m431]
cors={}
for c in allc:
    v=sub[c]
    if v.dtype==object: v=v.astype('category').cat.codes
    v=pd.to_numeric(v,errors='coerce').fillna(v.median() if v.dtype!=object else 0).values.astype(float)
    if np.std(v)>1e-9: cors[c]=float(np.corrcoef(v,r)[0,1])
cs=pd.Series(cors).sort_values()
print('NEG:'); print(cs.head(10).round(3)); print('POS:'); print(cs.tail(12).round(3))
# per-snapshot tail features
def qtr(col,q):
    return d.groupby('snapshot_day')[col].transform(lambda s: s.quantile(q))
d['t_s28_q90']=d.spend_28/qtr('spend_28',.9)
d['t_s84_q90']=d.spend_84/qtr('spend_84',.9)
d['t_s364_q95']=d.spend_364/qtr('spend_364',.95)
d['t_top5_s28']=(d.spend_28>=qtr('spend_28',.95)).astype(float)
d['t_top5_s364']=(d.spend_364>=qtr('spend_364',.95)).astype(float)
d['t_sq28']=np.sqrt(d.spend_28); d['t_sq84']=np.sqrt(d.spend_84)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_top5_s364','t_sq28','t_sq84']
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
tc={c:float(np.corrcoef(d.loc[m431,c].fillna(0).values,r)[0,1]) for c in tail}
print('tail residual corr:',pd.Series(tc).round(3).to_dict())
for name,grp in [('BASE',[]),('+EWMA',ew),('+TAIL',tail),('+EWMA+TAIL',ew+tail)]:
    p2=fitpred(prep(base_cols+grp,d),300,tr431,m431,y)
    print(name,'MAE@431',round(float(np.abs(p2-y[m431]).mean()),3))
# also check tail MAE on top decile
p3=fitpred(prep(base_cols+ew+tail,d),300,tr431,m431,y)
rr=pd.DataFrame({'y':y[m431],'p':p3}); rr['bin']=pd.qcut(rr.y,10,duplicates='drop')
print(rr.groupby('bin',observed=True).agg(n=('y','size'),ym=('y','mean'),pm=('p','mean'),mae=('y',lambda z:0)).drop(columns='mae').assign(
    mae=np.abs(rr.p.values-rr.y.values)).groupby('bin',observed=True).agg(n=('n','first'),ym=('ym','first'),pm=('pm','first'),mae=('mae','mean')).round(1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v2.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
base_cols=[c for c in A.load_saved('e013_stock.parquet').columns if c not in ('household_key','snapshot_day')]
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
def qtr(col,q): return d.groupby('snapshot_day')[col].transform(lambda s: s.quantile(q))
d['t_s28_q90']=d.spend_28/qtr('spend_28',.9); d['t_s84_q90']=d.spend_84/qtr('spend_84',.9)
d['t_s364_q95']=d.spend_364/qtr('spend_364',.95)
d['t_top5_s28']=(d.spend_28>=qtr('spend_28',.95)).astype(float)
d['t_top5_s364']=(d.spend_364>=qtr('spend_364',.95)).astype(float)
d['t_sq28']=np.sqrt(d.spend_28); d['t_sq84']=np.sqrt(d.spend_84)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_top5_s364','t_sq28','t_sq84']
# interaction candidates
d['i_pow']=d.spend_28**1.25
d['i_act']=d.spend_28*d.active_28
d['i_rec7']=d.spend_28*(d.recency<=7).astype(float)
d['i_ewact']=d.ew28*d.tp_act8 if 'tp_act8' in d else d.ew28
d['i_red']=d.spend_28*np.log1p(d.m_red_28)
d['i_gas']=d.spend_28*d.x_dep_KIOSK_GAS if 'x_dep_KIOSK-GAS' in d.columns else 0
d['i_misc']=d.spend_28*d['x_dep_MISC SALES TRAN']
inter=['i_pow','i_act','i_rec7','i_ewact','i_red','i_misc']
def prep(cols,frame):
    X=frame[cols].copy()
    for c in cols:
        if X[c].dtype==object: X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem,yy):
    nn=int(trm.sum()); Xd=np.hstack([np.ones((nn,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@yy[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
print('BASE+EWMA+TAIL:',round(float(np.abs(fitpred(prep(base_cols+ew+tail,d),300,tr431,m431,y)-y[m431]).mean()),3))
for c in inter:
    print('+'+c, round(float(np.abs(fitpred(prep(base_cols+ew+tail+[c],d),300,tr431,m431,y)-y[m431]).mean()),3))
print('+ALL inter:',round(float(np.abs(fitpred(prep(base_cols+ew+tail+inter,d),300,tr431,m431,y)-y[m431]).mean()),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v2.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
base_cols=[c for c in A.load_saved('e013_stock.parquet').columns if c not in ('household_key','snapshot_day')]
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
def qtr(col,q): return d.groupby('snapshot_day')[col].transform(lambda s: s.quantile(q))
d['t_s28_q90']=d.spend_28/qtr('spend_28',.9); d['t_s84_q90']=d.spend_84/qtr('spend_84',.9)
d['t_s364_q95']=d.spend_364/qtr('spend_364',.95)
d['t_top5_s28']=(d.spend_28>=qtr('spend_28',.95)).astype(float)
d['t_top5_s364']=(d.spend_364>=qtr('spend_364',.95)).astype(float)
d['t_sq28']=np.sqrt(d.spend_28); d['t_sq84']=np.sqrt(d.spend_84)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_top5_s364','t_sq28','t_sq84']
d['i_pow']=d.spend_28**1.25
d['i_act']=d.spend_28*d.active_28
d['i_rec7']=d.spend_28*(d.recency<=7).astype(float)
d['i_red']=d.spend_28*np.log1p(d.m_red_28)
d['i_misc']=d.spend_28*d['x_dep_MISC SALES TRAN']
inter=['i_pow','i_act','i_rec7','i_red','i_misc']
def prep(cols,frame):
    X=frame[cols].copy()
    for c in cols:
        if X[c].dtype==object: X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem,yy):
    nn=int(trm.sum()); Xd=np.hstack([np.ones((nn,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@yy[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
print('B+E+T:',round(float(np.abs(fitpred(prep(base_cols+ew+tail,d),300,tr431,m431,y)-y[m431]).mean()),3))
for c in inter:
    print('+'+c, round(float(np.abs(fitpred(prep(base_cols+ew+tail+[c],d),300,tr431,m431,y)-y[m431]).mean()),3))
print('+ALL inter:',round(float(np.abs(fitpred(prep(base_cols+ew+tail+inter,d),300,tr431,m431,y)-y[m431]).mean()),3))
