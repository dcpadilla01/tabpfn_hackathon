import agent_api as A, pandas as pd, numpy as np

def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z))] + [z[c].values.astype(float) for c in z.columns])
    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values.astype(float)), res(y.values.astype(float))
    return np.corrcoef(rx, ry)[0,1]

tt = A.train_targets()
X = A.build_features(lambda v, d: pd.DataFrame(index=pd.Index(v.households, name='household_key'), data={'z':1.0}))  # placeholder not needed; rebuild marketing features inline
