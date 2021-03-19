import torch
import numpy as np
import time


def cross_conv_corr(x, x_hat):
    if len(x.shape) != 3 or len(x_hat.shape) != 3:
        raise RuntimeError('Expects images with dimensions (batch_idx, nx, ny)')
    
    x_comp = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2)
    x_comp[:, :, :, 0] = x

    x_hat_comp = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2)
    x_hat_comp[:, :, :, 0] = x_hat

    fx = torch.fft(x_comp, signal_ndim=2)
    fx_hat = torch.fft(x_hat_comp, signal_ndim=2)

    cross_conv_fft = torch.stack([
                fx[:, :, :, 0] * fx_hat[:, :, :, 0] - fx[:, :, :, 1] * fx_hat[:, :, :, 1],
                fx[:, :, :, 0] * fx_hat[:, :, :, 1] + fx[:, :, :, 1] * fx_hat[:, :, :, 0]
    ], -1)
    cross_corr_fft = torch.stack([
                fx[:, :, :, 0] * fx_hat[:, :, :, 0] + fx[:, :, :, 1] * fx_hat[:, :, :, 1],
                fx[:, :, :, 0] * fx_hat[:, :, :, 1] - fx[:, :, :, 1] * fx_hat[:, :, :, 0]
    ], -1)

    cross_conv = torch.ifft(cross_conv_fft, signal_ndim=2)
    cross_corr = torch.ifft(cross_corr_fft, signal_ndim=2)
    return cross_conv[..., 0], cross_corr[... , 0]

def cosine_distance_invariant(x, x_hat, return_flip_index=False):
    cross_conv, cross_corr = cross_conv_corr(x, x_hat)
    max_abs_corr = torch.max(torch.abs(cross_corr).reshape(cross_corr.shape[0:1] + (-1,)), -1)
    max_abs_conv = torch.max(torch.abs(cross_conv).reshape(cross_conv.shape[0:1] + (-1,)), -1)

    dist = torch.max(max_abs_conv[0], max_abs_corr[0]) 

    if return_flip_index:
        flip = max_abs_conv[0] > max_abs_corr[0]  
        idx = []
        for im in range(x.shape[0]):
            if flip[im]:
                idx.append(torch.where(torch.abs(cross_conv)[im, ...] ==  max_abs_conv[0][im])) 
            else: 
                idx.append(torch.where(torch.abs(cross_corr)[im, ...] ==  max_abs_corr[0][im]))
        
        return dist, flip, idx
    else:
        return dist.to(x.device)

# FIXMEHL make sure things here should stay here
def get_beamstop_masks(xdim, ydim, approx_beamstop_area_frac=0.03, centered=False, shape='rectangular', x_pix=None, y_pix=None):
    # output will be ydim x xdim
    if shape == 'rectangular' and x_pix is None:
        xticks = np.linspace(-0.5, 0.5, xdim)
        yticks = np.linspace(0.5, -0.5, ydim)
        xp, yp = np.meshgrid(xticks, yticks)
        x_cutoff = np.sqrt(approx_beamstop_area_frac) / 2
        y_cutoff = x_cutoff
        beamstop_mask = np.logical_and(np.abs(xp) < x_cutoff, np.abs(yp) < y_cutoff)
        if not centered:
            beamstop_mask = np.fft.fftshift(beamstop_mask)
    else:
        if shape == 'square' and x_pix is None:
            # set x_pix, y_pix
            x_pix = np.sqrt((xdim * ydim * approx_beamstop_area_frac))
            y_pix = x_pix
        beamstop_mask = np.zeros((ydim, xdim), dtype='Bool')
        for i in range(round(x_pix / 2)): # was int; truncates. switch to round
            for j in range(round(y_pix / 2)):
                beamstop_mask[j, i] = True
                beamstop_mask[j, xdim - i - 1] = True
                beamstop_mask[ydim - j - 1, i] = True
                beamstop_mask[ydim - j - 1, xdim - i - 1] = True
        if centered:
            beamstop_mask = np.fft.fftshift(beamstop_mask)
            
    return beamstop_mask

