import agent_api as A, pandas as pd, numpy as np
E4 = A.load_saved("e004_temporal.parquet")
tt = A.train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w
F = E4.merge(tt.rename("y").reset_index(), on=["household_key","snapshot_day"], how="inner")
e4cols = [c for c in E4.columns if c not in ("household_key","snapshot_day")]
y = F.y.values
print("MAD of y (mean-abs dev from mean):", round(np.abs(y-y.mean()).mean(),2), " median:", np.median(y))

def ridge_fit(X, t, lam=1.0):
    mu=X.mean(0); sd=X.std(0)+1e-9
    Xn=(X-mu)/sd
    Xn=np.column_stack([np.ones(len(Xn)),Xn])
    M=Xn.T@Xn+lam*np.eye(Xn.shape[1]); M[0,0]-=lam
    return np.linalg.solve(M,Xn.T@t), mu, sd
def ridge_pred(X, beta, mu, sd):
    Xn=(X-mu)/sd
    return np.column_stack([np.ones(len(Xn)),Xn])@beta

Xb = np.column_stack([np.log1p(F[c].clip(lower=0).fillna(0)) for c in e4cols])
ly = np.log1p(y)
beta,mu,sd = ridge_fit(Xb, ly)
pred = np.expm1(ridge_pred(Xb,beta,mu,sd))
print("E004 log-ridge in-sample MAE:", round(np.abs(pred-y).mean(),2), " R2raw:", round(1-((pred-y)**2).sum()/((y-y.mean())**2).sum(),4))

# add decay84, decay180 recomputed per snapshot
dec = np.zeros(len(F))
dec180 = np.zeros(len(F))
tx_all = {d: A.snapshot(d).table("transactions") for d in sorted(F.snapshot_day.unique())}
pos = {d: sub.index for d, sub in F.groupby("snapshot_day")}
for d in sorted(F.snapshot_day.unique()):
    tx = tx_all[d]; tx=tx[tx.day<=d]
    age = (d-tx.day).astype(float)
    v84 = (tx.sales_value*np.exp(-age*np.log(2)/84)).groupby(tx.household_key).sum()
    v180 = (tx.sales_value*np.exp(-age*np.log(2)/180)).groupby(tx.household_key).sum()
    idx = pos[d]
    dec[idx] = F.loc[idx,"household_key"].map(v84).fillna(0).values
    dec180[idx] = F.loc[idx,"household_key"].map(v180).fillna(0).values
X2 = np.column_stack([Xb, np.log1p(dec), np.log1p(dec180)])
beta2,mu2,sd2 = ridge_fit(X2, ly)
pred2 = np.expm1(ridge_pred(X2,beta2,mu2,sd2))
print("E004+decay84+decay180 in-sample MAE:", round(np.abs(pred2-y).mean(),2), " R2raw:", round(1-((pred2-y)**2).sum()/((y-y.mean())**2).sum(),4))

# zero structure
print("\nzero-structure:")
for c in ["spend_28","active_28"]:
    z = (y==0)
    print(c, "P(y=0|feat=0):", round(z[F[c].fillna(0)==0].mean(),3), " P(y=0|feat>0):", round(z[F[c].fillna(-1)>0].mean(),3))
# oracle-ish: predict 0 where spend_28==0 else global mean of positive
m_pos = y[y>0].mean()
pr = np.where(F.spend_28.fillna(0).values==0, 0.0, m_pos)
print("MAE predict-0-if-spend28==0-else-mean:", round(np.abs(pr-y).mean(),2))
m_pos28 = y[(y>0)&(F.spend_28.fillna(0)>0)].mean()
pr2 = np.where(F.spend_28.fillna(0).values==0, 0.0, m_pos28)
print("MAE predict-0-if-spend28==0-else-mean(pos):", round(np.abs(pr2-y).mean(),2))

# market-level trailing spend per snapshot vs mean target
rows=[]
for d in sorted(F.snapshot_day.unique()):
    tx = tx_all[d]; t = tx[(tx.day>d-84)&(tx.day<=d)]
    rows.append((d, t.sales_value.sum()/max(1,t.household_key.nunique()), t.sales_value.sum()))
M = pd.DataFrame(rows, columns=["d","mkt84","mkt_tot"]).merge(F.groupby("snapshot_day").y.mean().rename("ymean"), left_on="d", right_index=True)
print("\nmarket84 vs snapshot mean target corr:", round(M.mkt84.corr(M.ymean),3))
print(M.round(1).to_string(index=False))
