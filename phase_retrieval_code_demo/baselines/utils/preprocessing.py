import torch
import numbers

def zero_pad(x, factor=2):
    b, c, h, w = x.shape
    
    if isinstance(factor, numbers.Integral):
        factor = factor, factor
    f1, f2 = factor
    
    n1, n2 = new_size = f1 * h, f2 * w
    p1, p2 = n1 - h, n2 - w
    
    return torch.nn.functional.pad(x, (0, p2, 0, p1))  # TORCH PADS IN F ORDER!!
    

def complexify(x):
    
    if x.shape[-1] != 2:
        x = torch.stack((x, torch.zeros_like(x)), axis=-1)
    return x


def complex_abs(x):
    return torch.sqrt((x ** 2).sum(-1))


def batch_cast_pad_and_fft(x):
    
    b, c, h, w = x.shape
    p1, p2 = padding = h, w
    
#    xp = torch.nn.functional.pad(x, (0, h, 0, w))
    xp = zero_pad(x)
#    xc = torch.stack((xp, torch.zeros_like(xp)), axis=-1)
    xc = complexify(xp)
    xf = torch.fft(xc, signal_ndim=2)
#    xm = torch.sqrt((xf ** 2).sum(-1))
    xm = complex_abs(xf)

    return xm

