import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Poisson
import torch.nn.functional as F
from phase_retrieval_code_demo.core.utils import get_beamstop_masks

class Obs():
    """
    Class to generate the holographic Coherent Diffraction Imaging observations
    """
    def __init__(self, oversampling=2,
                 args_noise={'noise_type': None},
                 args_ref={'method': 'binary', 'pad': 0},
                 args_beamstop={'beamstop_mask': None},
                 fill_locs=None, fill_values=None,
                 device='cpu'):
        
        self.device = device
        self.oversampling = oversampling
        if oversampling < 1:
            raise ValueError('Oversampling ratio should be at least 1')
        self.ref = False if args_ref['method'] is None else True
        if self.ref:
            self.x_ref = args_ref['x_ref'] if args_ref['method'] == 'custom' else None
            self.pad_ref = args_ref['pad']
            self.type_ref = args_ref['method']
            self.args_ref = args_ref

        for k, v in args_noise.items():
            setattr(self, k, v)

        self.beamstop_mask = args_beamstop['beamstop_mask']

        # for overriding beamstop magnitudes
        # (not zeroing out, as for unfilled beamstop)
        self.fill_locs = fill_locs
        self.fill_values = fill_values

    def pad_and_ref(self, x):
        """
        Fonction to prepocess the signal x:
        - adding dimension if x real to reprensent complex part
        - generating and adding a reference if needed 
        - padding with zeros according to the oversampling desired
        
        Parameters
        ==========

        x: tensor shape can either be 
            - (nim, nx, ny) (real)
            - (nim, 2, nx, ny) (complex)

        Returns
        ==========

        x_comp: tensor shape (nim, 2, nx_tot, ny_tot)
            where nx_tot, ny_tot include 
            signal, optional separation, optional reference and oversampling
        """
        try:
            nim, _, nx, ny = x.shape
        except:
            nim, nx, ny = x.shape

        if self.ref:
            ny_tot = int(self.oversampling * (2 + self.pad_ref) * ny) 
        else:
            ny_tot = int(self.oversampling * ny)
        nx_tot = int(self.oversampling * nx)
        
        x_comp = torch.zeros(x.shape[0], 2, nx_tot, ny_tot)

        if len(x.shape) == 4:
            x_comp[:, :, :nx, :ny] = x
        elif len(x.shape) == 3:
            x_comp[:, 0, :nx, :ny] = x

        if self.ref:
            if self.x_ref is None:
                if self.type_ref == 'binary':
                    self.x_ref = torch.randint(2, x.shape[-2:])
                elif self.type_ref == 'rand':
                    self.x_ref = torch.rand(x.shape[-2:])
                else:
                    raise ValueError('Ref type not understood')

                self.x_ref = self.x_ref.expand((nim, nx, ny))

            x_comp[:, 0, :nx, (1 + self.pad_ref) * ny:(2 + self.pad_ref) * ny] = self.x_ref

        return x_comp.to(self.device).double()

    def forward(self, x_comp, noise_off=False, 
                beamstop_if_avail=True, dofill=False):
        """
        Function simulating observations from padded signal (+ reference),
        observations are squared magnitudes of discrete Fourrier transform,
        
        Parameters
        ==========

        x_comp: tensor shape (nim, 2, nx_tot, ny_tot)
            output of obs.pad_and_ref(x, ...)
        
        noise_off: bool, True overrides noise parameters to get noiseless output
        
        beamstop_if_avail: FIXMEHL

        dofill: FIXMEHL

        Returns:
        ==========

        noisy_y: tensor shape (nim, nx_tot, ny_tot)
        """
        nim, _, nx_tot, ny_tot = x_comp.shape

        fx = torch.fft(x_comp.permute([0, 2, 3, 1]), signal_ndim=2)
        y = (fx**2).sum(-1)

        if noise_off or self.noise_type is None:
            noisy_y = y
        elif self.noise_type == 'gaussian':
            noisy_y = y + self.noise_sig * torch.randn(y.shape, device=y.device)
        elif self.noise_type == 'poisson':
            l1 = y.sum((-2,-1)).reshape(nim, 1, 1) 
            l1 /= (self.noise_n * nx_tot * ny_tot)
            rate = y  / l1
            noisy_y = l1 * Poisson(rate).sample()

        if self.beamstop_mask is not None and beamstop_if_avail:
            if isinstance(self.beamstop_mask, np.ndarray) or isinstance(self.beamstop_mask, torch.Tensor): 
                noisy_y[:, self.beamstop_mask] = 0 #self.beamstop_fill_values
            else: # assume it is a scalar size
                beamstop_dim = self.beamstop_mask
                sized_beamstop_mask = get_beamstop_masks(y.shape[-1], y.shape[-2], centered=False, shape='rectangular', x_pix=beamstop_dim + 1, y_pix=beamstop_dim + 1)
                noisy_y[:, sized_beamstop_mask] = 0

        if dofill and self.fill_locs is not None: # if we want to override some of the true magnitudes with "filled" values, e.g. on beamstop:
            final = torch.zeros(noisy_y.shape, dtype=noisy_y.dtype, device=noisy_y.device, requires_grad=False)

            final[:, self.fill_locs] = self.fill_values
            final[:, np.logical_not(self.fill_locs)] = noisy_y[:, np.logical_not(self.fill_locs)]
            return final

        return noisy_y