import numpy as np, pandas as pd, agent_api as api
t = api.load_saved('e003_catmix.parquet')
tt = api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
days = sorted(df.snapshot_day.unique())
print('snapshots', days, 'rows', len(df))
def eval_laso(feats, lam=100, log=False):
    mae=[]; res_all=[]; df_all=[]
    for d in days:
        tr = df.snapshot_day!=d; va = df.snapshot_day==d
        Xtr = df.loc[tr, feats].astype(float).values
        mu = np.nanmean(Xtr,0); Xtr=np.where(np.isnan(Xtr),mu,Xtr)
        ytr = df.loc[tr,'future_spend_4w'].values
        Xva = df.loc[va, feats].astype(float).values; Xva=np.where(np.isnan(Xva),mu,Xva)
        yva = df.loc[va,'future_spend_4w'].values
        sd=Xtr.std(0); sd[sd==0]=1
        Ztr=(Xtr-mu)/sd; Zva=(Xva-mu)/sd
        if log:
            ytr=np.log1p(ytr)
        w=np.linalg.solve(Ztr.T@Ztr+lam*np.eye(len(feats)), Ztr.T@ytr)
        pv=Zva@w
        if log: pv=np.expm1(pv)
        r=pv-yva; mae.append(np.abs(r).mean()); res_all.append(r); df_all.append(df.loc[va].assign(pred=pv, res=r, y=yva))
    return np.mean(mae), pd.concat(df_all)
m,_=eval_laso(feats); print('E003 proxy MAE %.3f'%m)
m2,_=eval_laso(feats, log=True); print('E003 log-target proxy MAE %.3f'%m2)
_,D=eval_laso(feats)
D['yb']=pd.cut(D.y,[-1,.5,50,150,300,700,1e9])
print(D.groupby('yb',observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean(),'bias':g.res.mean(),'mp':g.pred.mean()})))
print('\nby snapshot:'); print(D.groupby('snapshot_day').apply(lambda g: np.abs(g.res).mean()))