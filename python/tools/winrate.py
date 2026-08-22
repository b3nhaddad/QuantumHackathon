import numpy as np

def win_rate(a,b,strategy):
    results = strategy(a,b)
    np.unstack(results)
