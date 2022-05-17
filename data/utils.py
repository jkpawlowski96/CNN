import numpy as np

def shuffle(*arrays, seed=None):
    """
    Shuffle arrays, oprional by random seed
    """
    if seed:
        np.random.seed(seed)
    l = len(arrays[0])
    perm = np.random.permutation(l)
    res = []
    for array in arrays:
        res.append(array[perm])
    return tuple(res)
 
def limit_float(value, max_i=2, d=4):
    """
    Limit float decimal range for fine printing
    """
    max_s = max_i + 1 + d
    max_v = pow(10, max_i) - 1
    if value > max_v:
        if d > 1:
            return limit_float(value, max_i=max_i+1, d=d-1)
        else:
            pass
    v = np.round(value, d)
    if v > max_v:
        return 
    s = str(v)
    if '.' not in s:
        s = s + '.0'

    s_d = len(s.split('.')[-1])

    if s_d < d:
        _s_d = d - s_d
        s = s.ljust(len(s)+_s_d, '0')
    
    s = s.rjust(max_s, ' ')
    return s
