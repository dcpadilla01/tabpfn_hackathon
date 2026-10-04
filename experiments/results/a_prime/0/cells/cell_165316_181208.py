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
