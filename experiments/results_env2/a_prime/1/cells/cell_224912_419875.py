import numpy as np, pandas as pd
def probe(view, sd):
    out = pd.DataFrame({'x': np.arange(len(view.households))}, index=view.households)
    return out
feats = agent_api.build_features(probe)
print(type(feats), feats.shape)
print(feats.index[:5], feats.index.name)
print(feats.head(3))
print(feats.columns.tolist())
