import numpy as np
import torch

def complex_mult(x, y):
    
    m1 = x * y
    m2 = x * torch.flip(y, (-1,))
    
    real = m1[..., 0] - m1[..., 1]
    imag = m2.sum(-1)
    
    return torch.stack((real, imag), axis=-1)



def check_grid(grid_or_shape, normalize=True, center=True):
    
    if isinstance(grid_or_shape, tuple):
        slices = [slice(0, float(a)) for a in grid_or_shape]
        grid = np.mgrid[slices]
        
        if center:
            grid -= grid.mean(axis=tuple(range(1, grid.ndim)), keepdims=True)
        
        if normalize:
            grid /= np.abs(grid).max()
        
    else: # could add some more checks here
        grid = grid_or_shape
    
    return grid


