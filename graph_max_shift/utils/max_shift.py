import numpy as np

"""
Implementation of the (medoid) max shift method. At each iteration, move to the point within a radius r that maximizes the density.
Here the density is known.
"""

def one_step(x0, data , r, f):
    
    nbhd = np.where(np.linalg.norm(x0 - data, axis =1) <r)[0]
    
    next_idx = np.argmax(f(data[nbhd]))
    
    x1 = data[nbhd[next_idx]]
    
    return x1
                         
                         
def max_shift(x0, data, r , f):
    path = [x0]
    dist_traveled = np.inf
    
    while dist_traveled >0:
        x1 = one_step(x0, data , r, f)
        dist_traveled = np.linalg.norm(x0-x1)
        if dist_traveled > 0:
            path += [x1]
            x0 = x1
    return np.array(path)


