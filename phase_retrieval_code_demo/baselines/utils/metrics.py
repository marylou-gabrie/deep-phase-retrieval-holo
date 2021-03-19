import numpy as np
import torch

from .general import complex_mult
from .preprocessing import zero_pad, complexify
import numbers


def cross_corr_and_conv(x, y, pad=False, real=True):
    
    if pad:
        x = zero_pad(x)
        y = zero_pad(y)
    
    x = complexify(x)
    y = complexify(y)
    
    xf = torch.fft(x, signal_ndim=2)
    yf = torch.fft(y, signal_ndim=2)
    
    convf = complex_mult(xf, yf)
    
    yf_conj = yf
    yf_conj[..., 1] *= -1
    
    corrf = complex_mult(xf, yf_conj)
    
    conv = torch.ifft(convf, signal_ndim=2)
    corr = torch.ifft(corrf, signal_ndim=2)
    
    if real:
        conv = conv[..., 0]
        corr = corr[..., 0]
    
    return corr, conv


def invariant_euclidean_distance(x, y, conjugate_flip_invariant=True, sign_flip_invariant=True, 
                                 complex=False, pad=False, return_squared=False):
    if complex:
        raise NotImplementedError

    *b, h, w = x.shape
    x2 = torch.norm(x.reshape(b + [h * w]), dim=-1) ** 2
    
    *b, h, w = y.shape
    y2 = torch.norm(y.reshape(b + [h * w]), dim=-1) ** 2
    
    corr, conv = cross_corr_and_conv(x, y, pad=pad, real=~complex)
    
    if sign_flip_invariant:
        corr.abs_()
        conv.abs_()
    
    *b, h, w = corr.shape
    corr_max = corr.reshape(b + [h * w]).max(dim=-1)[0]
    conv_max = conv.reshape(b + [h * w]).max(dim=-1)[0]
    
    best = corr_max
    if conjugate_flip_invariant:
        best = torch.max(best, conv_max)
    
    distances_squared = x2 + y2 - 2 * best
    if return_squared:
        return distances_squared
    
    return torch.sqrt(torch.relu_(distances_squared))


def invariant_correlation(x, y, conjugate_flip_invariant=True, sign_flip_invariant=True, 
                                 complex=False, pad=False):
    if complex:
        raise NotImplementedError

    *b, h, w = x.shape
    x_norm = torch.norm(x.reshape(b + [h * w]), dim=-1)
    
    *b, h, w = y.shape
    y_norm = torch.norm(y.reshape(b + [h * w]), dim=-1)
    
    corr, conv = cross_corr_and_conv(x, y, pad=pad, real=~complex)
    
    if sign_flip_invariant:
        corr.abs_()
        conv.abs_()
    
    *b, h, w = corr.shape
    corr_max = corr.reshape(b + [h * w]).max(dim=-1)[0]
    conv_max = conv.reshape(b + [h * w]).max(dim=-1)[0]
    
    best = corr_max
    if conjugate_flip_invariant:
        best = torch.max(best, conv_max)
    
    correlations = best / (x_norm * y_norm)
    return correlations


def compute_shift_and_flip(x, y, pad=False, complex=False):
    if complex:
        raise NotImplementedError
    
    corr, conv = cross_corr_and_conv(x, y, pad=pad, real=~complex)
    
    *b, h, w = corr.shape
    corr_max, corr_argmax = torch.abs(corr.reshape(b + [h * w])).max(axis=-1)
    conv_max, conv_argmax = torch.abs(conv.reshape(b + [h * w])).max(axis=-1)
    
    corr_max_with_sign = corr.reshape((-1, h * w))[torch.arange(corr_argmax.numel()), corr_argmax.reshape(-1)].reshape(corr_argmax.shape)
    conv_max_with_sign = conv.reshape((-1, h * w))[torch.arange(conv_argmax.numel()), conv_argmax.reshape(-1)].reshape(corr_argmax.shape)
    corr_sign = 2 * (corr_max_with_sign > 0) - 1
    conv_sign = 2 * (conv_max_with_sign > 0) - 1
    
    val, flip = torch.stack((corr_max, conv_max), axis=0).max(0)
    sign = torch.gather(torch.stack((corr_sign, conv_sign), axis=0), 0, torch.stack((flip, 1-flip), axis=0))[0]
    
    argmax = torch.stack((corr_argmax, conv_argmax), axis=0)[flip, 
                                                             torch.arange(len(flip))[:, np.newaxis],
                                                             torch.arange(flip.shape[1])]
    I, J = argmax // w, argmax % w
    
    return I, J, flip, val, sign


def shift_and_flip_images(x, shift_I, shift_J, flip, sign=None, complex=False):
    if complex:
        raise NotImplementedError
    
    if sign is None:
        sign = torch.ones_like(flip)
    
    output = x.clone()
    *b, h, w = output.shape
    output_raveled = output.view(np.prod(b), h, w)
    
    for out, si, sj, f, s in zip(output_raveled,
                              shift_I.reshape(-1),
                              shift_J.reshape(-1),
                              flip.reshape(-1),
                              sign.reshape(-1)):
        if f:
            out[:] = torch.flip(out, (0, 1))
            si = si + 1
            sj = sj + 1
        out[:] = torch.roll(out, (si, sj), (0, 1)) * s
        
    return output

        
def register_images(x, y, pad=False, complex=False):
    if complex:
        raise NotImplementedError
    
    I, J, flip, val, sign = compute_shift_and_flip(y, x, pad=pad, complex=complex)
    
    registered = shift_and_flip_images(x, I, J, flip, sign)
    
    return registered


def registered_distance(x, y, pad=False, complex=False, squeeze=False):
    if complex:
        raise NotImplementedError
        
    if len(x.shape) != len(y.shape):
        print('''
        from registered_distance(x, y) in phase_retrieval_code_demo.michael.utils.general: 
        x.shape (= {}) and y.shape (= {}) do not match. registered_distance 
        will broadcast shapes accordingly, but this may lead to unexpected behavior!
        '''.format(x.shape, y.shape))
            
    
    registered_x = register_images(x, y, pad=pad, complex=complex)
    differences = y - registered_x
    
    *b, h, w = differences.shape
    distances = torch.norm(differences.reshape(b + [h * w]), dim=-1)
    
    return distances

    
    
    
