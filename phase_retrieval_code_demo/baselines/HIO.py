import numpy as np
import torch

from .utils.preprocessing import complex_abs, zero_pad
from .utils.metrics import registered_distance
import matplotlib.pyplot as plt


def _support_projector(x, non_support="not_upper_left_quarter", real=False, positive=False, inplace=True, idx=None, return_idx=False):
    """Projects orthogonally onto support.
    
    Parameters
    ==========
    
    x: ndarray, shape (n_imgs, n_channels, h, w, 2)
        Input array to be projected
        
    non_support: ndarray, shape (h, w), dtype bool
        False on support, True outside support
        If not specified, support is assumed to be top left quarter
        
    real: bool, default False
        project values onto reals
    
    positive: bool, default False
        project values onto positive orthant
    """
    
    if not inplace or return_idx:
        x_ = x
        x = x.clone()
        
    if idx is not None:
        x[idx] = 0
    
    if non_support is "not_upper_left_quarter":
        h, w = x.shape[-3:-1]
        x[..., h//2:, :, :] = 0.
        x[..., :, w//2:, :] = 0.
    else:
        x[:, :, non_support] = 0.
    
    if real:
        x[..., 1] = 0
    
    if positive:
        x = torch.relu_(x)
        
    if return_idx:
        idx = torch.logical_not(torch.isclose(x, x_))
        return x, idx
    else:
        return x


def _magnitude_projector(x, magnitudes, compute_residual=False, enforce=None):
    """Projects on to Fourier magnitude torus
    
    Parameters
    ==========
    x: ndarray, shape (n_imgs, n_channels, h, w, 2)
    
    magnitudes: ndarray, shape (n_imgs, h, w)
    
    compute_residual: bool, default False
        computes the discrepancy between magnitudes of x and desired magnitudes

    enforce: ndarray of type bool, default None (enforces all magnitudes)
        which magnitudes to enforce
    """
    
    if enforce is None:
        enforce = np.ones((x.shape[2], x.shape[3])).astype('bool')
    
    xf = torch.fft(x, signal_ndim=2)
    xfm = torch.sqrt((xf ** 2).sum(-1, keepdims=True))
    xf[:, :, enforce, :] /= (xfm[:, :, enforce, :] + 1e-19)


    temp = magnitudes[:, np.newaxis, enforce]
    xf[:, :, enforce, :] *= temp[:, :, :, np.newaxis]

    
    ixf = torch.ifft(xf, signal_ndim=2)
    if compute_residual:
        residual = torch.norm((xfm[..., 0] - magnitudes[:, np.newaxis]).reshape(x.shape[:2] + (-1,)), dim=-1)
        return ixf, residual
    return ixf


# def HIO1(magnitudes, n_inits=1, n_iter=10000, beta=1, positive=False, compute_magnitude_residual=True, non_support="not_upper_left_quarter", enforce=None):
#     """Performs Hybrid Input Output Algorithm in parallel.

#     Parameters
#     ==========

#     magnitudes: ndarray, shape=(n_imgs, height, width)
#         Oversampled magnitude images as input.

#     n_inits: int, default 1
#         Indicates how many instances of HIO to run per input image

#     n_iter: int, default 10000
#         Number of iterations to run HIO

#     beta: float, default 1
#         projection mixing parameter, regulates mixing outside the support
    
#     positive: bool, default False
#         Indicates whether a positivity constraint should be used

#     compute_magnitude_residual: bool, default True
#         Whether to return a time series of residuals

#     Returns
#     =======
#     reconstructions: ndarray, shape=(n_imgs, n_inits, h, w, 2)

#     residuals: ndarray, shape=(n_iter, n_imgs, n_inits)
#         Returned iff compute_magnitude_residual is True
# """    
#     n_imgs, h, w = magnitudes.shape
    
#     x = torch.randn(n_imgs, n_inits, h, w, 2).to(magnitudes.device)
#     x = _support_projector(x, non_support=non_support)

#     residuals = []    
#     for i in range(n_iter):
        
#         if compute_magnitude_residual:
#             y, residual = _magnitude_projector(x, magnitudes, compute_residual=True, enforce=enforce)
#             residuals.append(residual.detach().cpu().numpy())
#         else:
#             y = _magnitude_projector(x, magnitudes)
#         projected_y = _support_projector(y, inplace=False, non_support=non_support)
#         out_of_support_y = y - projected_y
#         projected_x = _support_projector(x, inplace=False, non_support=non_support)
#         out_of_support_x = x - projected_x
        
#         if positive:
#             #projected_y[projected_y < 0.] = 0.
#             torch.relu_(projected_y)
#         x = projected_y + out_of_support_x - beta * out_of_support_y
    
#     if compute_magnitude_residual:
#         return x, np.array(residuals)
#     return x



# def HIO2(magnitudes, non_support="not_upper_left_quarter", n_inits=1, n_iter=10000, beta=1., init_imgs=None):
#     """David's HIO algorithm"""
#     n, h, w = magnitudes.shape
#     if init_imgs is None:
#         init_imgs = torch.randn((n, n_inits, h, w, 2), device=magnitudes.device)
#         _support_projector(init_imgs, inplace=True)
    
#     # use phase from init img
#     x_init = _magnitude_projector(init_imgs, magnitudes, enforce=enforce)
#     x = x_prev = x_init
#     x_prev_out_of_support = x - _support_projector(x, non_support=non_support, inplace=False)
#     resids = []
#     for i in range(1, n_iter):
#         x = _magnitude_projector(x, magnitudes, enforce=enforce)
#         x_on_support = _support_projector(x, non_support=non_support, inplace=False)
#         x_out_of_support = x - x_on_support
#         x_prev_out_of_support = x_prev_out_of_support - beta * x_out_of_support
#         x = x_on_support + x_prev_out_of_support
#         resid = torch.norm(_support_projector(x - x_prev)[..., 0].reshape(n, n_inits, h * w), dim=-1).detach().cpu().numpy()
#         resids.append(resid)
#         x_prev = x
#     return x, np.array(resids)



# def HIO3(magnitudes, n_inits=1, n_iter=10000, beta=1, real=True, positive=False,
#          compute_magnitude_residual=True, init_imgs=None, non_support="not_upper_left_quarter", enforce=None):
#     """Performs Hybrid Input Output Algorithm in parallel.

#     Parameters
#     ==========

#     magnitudes: ndarray, shape=(n_imgs, height, width)
#         Oversampled magnitude images as input.

#     n_inits: int, default 1
#         Indicates how many instances of HIO to run per input image

#     n_iter: int, default 10000
#         Number of iterations to run HIO

#     beta: float, default 1
#         projection mixing parameter, regulates mixing outside the support
    
#     real: bool, default True
#         Indicates whether at every iteration the signal should be projected
#         onto the real numbers

#     positive: bool, default False
#         Indicates whether a positivity constraint should be used

#     compute_magnitude_residual: bool, default True
#         Whether to return a time series of residuals

#     init_imgs: ndarray, shape (n_imgs, n_inits, h, w, 2)
#         Initialization images in signal space. Random if None specified

#     non_support, generically: ndarray, shape (h, w)
#         Pixels that are outside the known support
#         Or shortcut, "not_upper_left_quarter" 

#     enforce: ndarray, shape (h, w)
#         Which magnitudes to enforce at magnitude projection step. Defaults 
#         to "None", which enforces all magnitudes
        
#     Returns
#     =======
#     reconstructions: ndarray, shape=(n_imgs, n_inits, h, w, 2)

#     residuals: ndarray, shape=(n_iter, n_imgs, n_inits)
#         Returned iff compute_magnitude_residual is True
# """    
#     n_imgs, h, w = magnitudes.shape
    
#     if init_imgs is None:
        
#         x = torch.randn(n_imgs, n_inits, h, w, 2).to(magnitudes.device)
#     else:
#         x = init_imgs

#     x = _support_projector(x, real=real, positive=positive)

#     residuals = []
#     for i in range(n_iter):
        
#         if compute_magnitude_residual:
#             y, residual = _magnitude_projector(x, magnitudes, compute_residual=True, enforce=enforce)
#             residuals.append(residual.detach().cpu().numpy())
#         else:
#             y = _magnitude_projector(x, magnitudes, enforce=enforce)
#         projected_y = _support_projector(y, inplace=False, real=real, positive=positive, non_support=non_support)
#         out_of_support_y = y - projected_y
#         projected_x = _support_projector(x, inplace=False, real=real, positive=positive, non_support=non_support)
#         out_of_support_x = x - projected_x
        
#         x = projected_y + out_of_support_x - beta * out_of_support_y
    
#     if compute_magnitude_residual:
#         return x, np.array(residuals)
#     return x

# def HIO4(magnitudes, n_inits=1, n_iter=10000, beta=1, real=True, positive=False,
#          compute_magnitude_residual=True, init_imgs=None, non_support="not_upper_left_quarter", enforce=None, ref_dict=None):
#     """
#     Minor change to HIO3:
    
#     projected_x and out_of_support_x are now projected with respect to the projection coordinates of y
#     """
#     n_imgs, h, w = magnitudes.shape
    
#     if init_imgs is None:
#         x = torch.randn(n_imgs, n_inits, h, w, 2).to(magnitudes.device)
#     else:
#         x = init_imgs

#     x = _support_projector(x, real=real, positive=positive, non_support=non_support)

#     residuals = []
#     for i in range(n_iter):
        
#         if compute_magnitude_residual:
#             y, residual = _magnitude_projector(x, magnitudes, compute_residual=True, enforce=enforce)
#             residuals.append(residual.detach().cpu().numpy())
#         else:
#             y = _magnitude_projector(x, magnitudes, enforce=enforce)
#         projected_y, idx = _support_projector(y, inplace=False, real=real, positive=positive, return_idx=True, non_support=non_support)
#         out_of_support_y = y - projected_y
#         projected_x = _support_projector(x, inplace=False, real=real, positive=positive, idx=idx, non_support=non_support)
#         out_of_support_x = x - projected_x
        
#         x = projected_y + out_of_support_x - beta * out_of_support_y
    
#         if ref_dict is not None: # holographic case: enforce known reference
#             if real: #reference assumed real 
#                 msk = torch.stack((ref_dict['ref_mask'], ref_dict['ref_mask']), dim=2)
#                 msk[..., 1] = False
#                 x[:, :, msk] = ref_dict['ref_vals'].view(1, 1, -1).repeat(1, x.shape[1], 1)
#             else:
#                 msk = torch.stack((ref_dict['ref_mask'], ref_dict['ref_mask']), dim=2)
#                 x[:, :, msk] = ref_dict['ref_vals'].view(1, 1, -1).repeat(1, x.shape[1], 1)

#     if compute_magnitude_residual:
#         return x, np.array(residuals)
#     return x


def HIO(magnitudes, n_inits=1, n_iter=10000, beta=1, real=True, positive=False,
         init_imgs=None, track_errors=False, ground_truth=None, non_support="not_upper_left_quarter", enforce=None, ref_dict=None, ref_imag_is_zero=False):
    """Performs Hybrid Input Output Algorithm in parallel.

    Parameters
    ==========

    magnitudes: ndarray, shape=(n_imgs, height, width)
        Oversampled magnitude images as input.

    n_inits: int, default 1
        Indicates how many instances of HIO to run per input image

    n_iter: int, default 10000
        Number of iterations to run HIO

    beta: float, default 1
        projection mixing parameter, regulates mixing outside the support
    
    real: bool, default True
        Indicates whether at every iteration the signal should be projected
        onto the real numbers

    positive: bool, default False
        Indicates whether a positivity constraint should be used

    compute_magnitude_residual: bool, default True
        Whether to return a time series of residuals

    init_imgs: ndarray, shape (n_imgs, n_inits, h, w, 2)
        Initialization images in signal space. Random if None specified
    
    track_errors: bool, default False
        Keeps track of the magnitude error and the signal error if ground truth is provided
    
    ground_truth: ndarray, shape (n_imgs, img_height, img_width), default None
        ground-truth images that are compared using registered_distance

    enforce: ndarray, shape (h, w)
        Which magnitudes to enforce at magnitude projection step. Defaults 
        to "None", which enforces all magnitudes

    refdict: dictionary for holographic case
        refdict['ref_mask']: bool array (torch or numpy), shape (h,w)
            True where the reference is located 
        refdict['ref_vals']: array with total # entries = # True values in refdict['ref_mask'] if real is True; else twice that
            Reference values, which will be enforced after each iteration
            (2x the number of values if real==True)

    ref_imag_is_zero: bool to enforce that the reference's imaginary component is zero, when real is True
        Defaults to True
        
    Returns
    =======
    reconstructions: ndarray, shape=(n_imgs, n_inits, h, w, 2)

    residuals: ndarray, shape=(n_iter, n_imgs, n_inits)
        Returned iff compute_magnitude_residual is True
"""    
    n_imgs, h, w = magnitudes.shape
    
    if init_imgs is None:
        
        x = torch.randn(n_imgs, n_inits, h, w, 2).to(magnitudes.device)
    else:
        x = init_imgs

    x = _support_projector(x, real=real, positive=positive, non_support=non_support)

    magnitude_residuals = []
    signal_residuals = []

    if ground_truth is not None:
        if ground_truth.shape != magnitudes.shape:
            ground_truth = zero_pad(ground_truth)
    
    for i in range(n_iter):
        
        y = _magnitude_projector(x, magnitudes, enforce=enforce)
        projected_y = _support_projector(y, inplace=False, real=real, positive=positive, non_support=non_support)
        out_of_support_y = y - projected_y
        projected_x = _support_projector(x, inplace=False, real=real, positive=positive, non_support=non_support)
        out_of_support_x = x - projected_x
        
        x = projected_y + out_of_support_x - beta * out_of_support_y
        
        if ref_dict is not None: # holographic case: enforce known reference
            if real: #reference assumed real 
                msk = torch.stack((ref_dict['ref_mask'], ref_dict['ref_mask']), dim=2)
                msk[..., 1] = False
                x[:, :, msk] = ref_dict['ref_vals'].view(1, 1, -1).repeat(1, x.shape[1], 1)
                if ref_imag_is_zero:
                    msk_for_imag = torch.stack((ref_dict['ref_mask'], ref_dict['ref_mask']), dim=2)
                    msk_for_imag[..., 0] = False
                    x[..., msk_for_imag] = 0 # set imaginary part to 0, since reference is real
            else:
                msk = torch.stack((ref_dict['ref_mask'], ref_dict['ref_mask']), dim=2)
                x[:, :, msk] = ref_dict['ref_vals'].view(1, 1, -1).repeat(1, x.shape[1], 1)

        if track_errors:
            x_mag = complex_abs(torch.fft(projected_x, signal_ndim=2))
            mag_diff = x_mag - magnitudes[:, np.newaxis]
            mag_dist = torch.norm(mag_diff.reshape(n_imgs, n_inits, -1), dim=2)
            magnitude_residuals.append(mag_dist.detach().cpu().numpy())
            
            if ground_truth is not None:
                if len(ground_truth.shape) == len(projected_x[..., 0].shape) - 1:
                    print('from HIO5(): ground_truth.shape (= {}) may be missing a dimension! (for reference, hio iterate shape is {}) adding extra dimension...'.format(ground_truth.shape, projected_x[..., 0].shape))
                    ground_truth = ground_truth.unsqueeze(1)
                    print('new ground_truth.shape: {}'.format(ground_truth.shape))
                signal_dist = registered_distance(projected_x[..., 0], ground_truth, pad=False)
                signal_residuals.append(signal_dist.detach().cpu().numpy())
    if track_errors:
        return x, np.array(magnitude_residuals), np.array(signal_residuals)
    return x
