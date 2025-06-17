import numpy as np

def min_max_scale(X):
    mins_x=np.min(X,axis=0)
    maxs_x=np.max(X,axis=0)
    X_st=(X-mins_x)/(maxs_x-mins_x)
    return X_st