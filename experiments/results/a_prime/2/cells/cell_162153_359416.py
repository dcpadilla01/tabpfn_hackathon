import agent_api, pandas as pd, numpy as np
def probe(view, s):
    tx = view.transactions
    hh = np.asarray(view.households).ravel().tolist() if view.households is not None else tx.household_key.unique()
    fd = tx.groupby('household_key')['day'].min()
    n_elig = int((fd<=s-84).sum()); n_hh = len(fd)
    print(f's={s}: households={len(hh)} tx={len(tx)} first<=s-84: {n_elig}/{n_hh}')
    return pd.DataFrame(index=hh)

out = agent_api.build_features(probe)
print('done', out.shape)
