import numpy as np
import torch
from .utils.preprocessing import complexify
from .utils.general import complex_mult


def holo_forward(X, R, Np=np.inf, random_seed=0):
    assert X.shape == R.shape
    X0R = np.hstack([X, np.zeros_like(X), R])
    s0, s1 = X0R.shape
    X0R_padded = np.pad(X0R, [(0, s0), (0, s1)])
    
    fX0R_padded = np.fft.fftshift(np.fft.fft2(X0R_padded))
    m2fX0R_padded = np.abs(fX0R_padded) ** 2
    
    if Np == np.inf:
        return m2fX0R_padded
    
    rng = np.random.RandomState(random_seed)
    l1 = m2fX0R_padded.sum()
    
    return rng.poisson(Np / l1 * m2fX0R_padded) * l1 / Np


def wiener_filt_numpy(Y, R, Np):
    
    W = np.fft.ifft2(Y)
    s1, s2 = R.shape
    s2 *= 3
    
#     i_indices = np.array(list(range(-s1, 0)) + list(range(0, s1)))[:, np.newaxis]
#     j_indices = list(range(-s2, 0)) + list(range(s2))
#     S_auto = W[i_indices, j_indices]

    S_auto = np.fft.fftshift(W)

    XR_cross = S_auto[:, :2 * R.shape[1]]
    t1, t2 = XR_cross.shape
    F_R = np.fft.fft2(R, (t1, t2))
    
    F_SNR = Np * np.linalg.norm(Y, 'fro') ** 2 / np.linalg.norm(Y, ord=1) ** 2
    
    X_ = np.fft.ifft2(np.fft.fft2(XR_cross, (t1, t2)) / np.conj(F_R) * np.abs(F_R) ** 2 / (np.abs(F_R) ** 2 + 1. / F_SNR))
    X = X_[R.shape[0]:, R.shape[1]:]
    return X#, X_, S_auto, XR_cross


def torch_fft_shift(x, dims):
    shifts = tuple(np.array([x.shape[dim] for dim in dims]) // 2)

    return torch.roll(x, shifts, dims)


def wiener_filt_torch(Y, R, Np, batch_dim=False):
    if not batch_dim:
        Y, R = Y.unsqueeze(0), R.unsqueeze(0)
        
    Yc = complexify(Y)
    W = torch.ifft(Yc, signal_ndim=2)

    n, s1, s2 = R.shape
    s2 *= 3
    
    S_auto = torch_fft_shift(W, dims=(1, 2))
    XR_cross = S_auto[:, :, :2 * R.shape[1]]

    _, t1, t2, _ = XR_cross.shape

    R_ = torch.zeros((n, t1, t2, 2), dtype=torch.float, device=W.device)
    R_[:, :s1, :R.shape[2], 0] = R
    F_R = torch.fft(R_, signal_ndim=2)
    F_R_conj = torch.clone(F_R)
    F_R_conj[..., 1] *= -1
    F_R_abs = (F_R ** 2).sum(-1, keepdim=True)

    F_SNR = Np * torch.norm(Y, dim=(1,2)) ** 2 / torch.norm(Y, p=1, dim=(1,2)) ** 2
    print('XR_cross.shape!', XR_cross.shape, torch.norm(Y, dim=(1,2)).shape)
    F_SNR = F_SNR.reshape(n, 1, 1, 1)
    X_ = torch.ifft(complex_mult(torch.fft(XR_cross, signal_ndim=2), F_R) / (F_R_abs + 1 / F_SNR), signal_ndim=2)
    print('X_.shape', X_.shape, R.shape[1], R.shape[2])

    X = X_[:, R.shape[1]:, R.shape[2]:]
    
    if not batch_dim:
        X = X[0]
        
    return X

def inv_filt_numpy(Y, R):
    
    W = np.fft.ifft2(Y)
    s1, s2 = R.shape
    s2 *= 3
    
    S_auto = np.fft.fftshift(W)

    XR_cross = S_auto[:, :2 * R.shape[1]]
    t1, t2 = XR_cross.shape
    F_R = np.fft.fft2(R, (t1, t2))
        
    X_ = np.fft.ifft2(np.fft.fft2(XR_cross, (t1, t2)) / np.conj(F_R))
    X = X_[R.shape[0]:, R.shape[1]:]
    return X


def inv_filt_torch(Y, R, batch_dim=False):
    if not batch_dim:
        Y, R = Y.unsqueeze(0), R.unsqueeze(0)
        
    Yc = complexify(Y)
    W = torch.ifft(Yc, signal_ndim=2)

    n, s1, s2 = R.shape
    s2 *= 3

    S_auto = torch_fft_shift(W, dims=(1, 2))
    XR_cross = S_auto[:, :, :2 * R.shape[1]]

    _, t1, t2, _ = XR_cross.shape

    R_ = torch.zeros((n, t1, t2, 2), dtype=torch.float, device=W.device)
    R_[:, :s1, :R.shape[2], 0] = R
    F_R = torch.fft(R_, signal_ndim=2)
    F_R_conj = torch.clone(F_R)
    F_R_conj[..., 1] *= -1
    F_R_abs = (F_R ** 2).sum(-1, keepdim=True)

    X_ = torch.ifft(complex_mult(torch.fft(XR_cross, signal_ndim=2), F_R) / F_R_abs, signal_ndim=2)

    X = X_[:, R.shape[1]:, R.shape[2]:]
    
    if not batch_dim:
        X = X[0]
        
    return X
