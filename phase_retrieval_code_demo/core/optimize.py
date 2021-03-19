import copy
import math 
import torch
import numpy as np
import matplotlib.pyplot as plt
from .forward_model import Obs
from .prior_model import NoPrior, DDPrior
from .utils import cosine_distance_invariant

def poisson_loss(y, y_hat):
    eps = 1e-10
    inds_y_nonzero = (y_hat != 0).type(y.dtype)
    pl = (y_hat * inds_y_nonzero - y * inds_y_nonzero * torch.log(y_hat + eps))
    return (0.5 * pl).sum(axis=[-2, -1])

def squared_loss(y, y_hat):
    return ((y - y_hat) ** 2).sum((-2, -1))

def fit(y, obs, x_true, niter=10,
        args_opt={'method': 'Adam', 'lr': 0.01},
        args_prior={'method': None},
        loss_type='squared',
        verbose_splits=10,
        display=True,
        device='cpu'):
    """
    Function implementing holo-P-opt and holo-s-opt 
    (with optional deep decoder)
    The optimization can be run for several images in parallel

    Parameters
    ==========
    
    y: tensor, shape (nim, nx, ny) 
        observations coming from obs.forward(x_comp, ...)

    obs: observation class instance from forward_model.py
        model for observations assumed for reconstruction
    
    x_true: tensor, shape (nim, nx, ny) if real 
        or shape (nim, 2, nx, ny) if complex
        ground truth signal used to estimate errors
        Note: for complex reconstruction x_true must be 4d 
              otherwise enforce real

    niter: int, number of iteration of optimizer

    loss_type: str, either 'squared' or 'poisson'

    verbose_splits: int, number of print errors along iterations

    display: bool, if True plots reconstruction for each "verbose split"

    FIXMEML For args_ parameter see detailed doc.

    Returns
    =======

    losses_y: ndarray shape (niter,)
        values of objective function throughout optimization

    resds_y: ndarray shape (niter, nim)
        values of squared residual errors on observations

    losses_x: ndarray shape (niter, nim)
        evaluations of squared errors 
        (for precise number, recompute from x_best at the end)

    x_splits: dic 
        x_splits['t']: list of iteration numbers
        x_plits['x']: list of x arrays, 
            reconstruction at corresponding iterations

    x_best: ndarray shape = same as x_true
        reconstructed signal at minimum value of residual error
        encountered during optim

    best_dd: instance of deepdecoder model
        when parameters of prior allow, 
        deep decoder model corresponding to x_best
    """
    if len(x_true.shape) == 2:
        raise RuntimeError("True signal should be 3 or 4d array - " 
                           "first dim corresponds to index of sample in batch -"
                           "optional second dim correspondis to real-im parts")
    elif len(x_true.shape) == 3:
        domain = 'real'
    else:
        domain = 'complex'

    if loss_type == 'squared':
        loss_fn = squared_loss
    elif loss_type == 'poisson':
        loss_fn = poisson_loss
    else:
        raise RuntimeError('Loss type not understood')

    def loss(y_hat, x_hat):
        return loss_fn(y, y_hat).sum()

    if args_prior['method'] is None:
        x_init = None
        init_mode = None
        if 'x_init' in args_prior.keys():
            x_init = args_prior['x_init']
        else:
            if 'init_mode' in args_prior.keys():
                init_mode = args_prior['init_mode']
            else:
                init_mode = 'randn'
        prior = NoPrior(x_true.shape, device=device, 
                        x_init=x_init, init_mode=init_mode)
        params = [prior.x_hat]
    elif args_prior['method'] == 'deepdecoder':
        if 'init' in args_prior.keys():
            prior = DDPrior(x_true.shape, device=device, 
                            init=args_prior['init'])
        else:
            if 'depth' in args_prior.keys():
                depth = args_prior['depth']
            else:
                depth = 2
            if 'channels' in args_prior.keys():
                channels = args_prior['channels']
            else:
                channels = 64
            if 'need_sigmoid' in args_prior.keys():
                need_sigmoid = args_prior['need_sigmoid']
            else:
                need_sigmoid = True
            prior = DDPrior(x_true.shape, device=device, 
                            num_channels_up=[channels]*depth,
                            need_sigmoid=need_sigmoid)
        params = prior.params
    else:
        raise RuntimeError('Prior not understood')

    if args_opt['method'] == 'LBFGS':
        optimizer = torch.optim.LBFGS(params=params, lr=args_opt['lr'])
    elif args_opt['method'] == 'Adam':
        optimizer = torch.optim.Adam(params=params, lr=args_opt['lr'])
    elif args_opt['method'] == 'SGD':
        optimizer = torch.optim.SGD(params=params, lr=args_opt['lr'])
    else:
        raise RuntimeError('Optimizer not understood')

    losses_y = []  #keeps track of global loss of the optimization
    resds_y = []  #keeps track of normalized squared residuals for each image
    losses_x = []
    x_best = prior.x_hat.data
    x_splits = {"t": [], "x": []}
    best_resd = np.inf * torch.ones(x_true.shape[0]).to(device).double()

    if display:
        plt.figure(figsize=(10, 3))
        plt.clf()
        cols = 5
        rows = math.ceil(verbose_splits / cols)

    x_init_comp = obs.pad_and_ref(prior.x_hat.clone())
    y_init = obs.forward(x_init_comp, noise_off=True)
    loss_y_init = loss(y_init, x_init_comp)
    print('Loss at initialization: {:e}'.format(loss_y_init.item()))

    for t in range(niter):
        def closure():
            optimizer.zero_grad()

            if args_prior['method'] is not None:
                prior.x_hat = prior.decode(prior.z_hat)

            x_hat_local = obs.pad_and_ref(prior.x_hat)
            
            y_hat = obs.forward(x_hat_local, noise_off=True)

            loss_y = loss(y_hat, x_hat_local)

            loss_y.backward(retain_graph=True) 

            resd_y = torch.sqrt(((y - y_hat) ** 2).sum((-2, -1))) 
            resd_y /= torch.sqrt((y ** 2).sum((-2, -1))) 
            resds_y.append(resd_y.detach().cpu())
            losses_y.append(loss_y.detach().cpu().numpy())

            return loss_y

        optimizer.step(closure)

        if obs.ref:
            loss_x = ((x_true.double() - prior.x_hat.double()) ** 2).sum((-2, -1))
        else:
            # computes errors invariant to flip and translations 
            # which is only necessary without a reference 
            if domain == 'complex':
                mod_x = torch.sqrt((x_true.double() ** 2).sum(1))
                mod_xhat = torch.sqrt((prior.x_hat.double() ** 2).sum(1)).cpu()
            else:
                mod_x = x_true.double()
                mod_xhat = prior.x_hat.double()

            loss_x = (mod_x ** 2).sum((-2, -1)) + (mod_xhat ** 2).sum((-2, -1))
            loss_x -= - 2 * cosine_distance_invariant(mod_x, mod_xhat).double()

        losses_x.append(loss_x.detach().cpu().numpy())

        freq_splits = max(int(niter / verbose_splits), 1)
        if (t+1) % freq_splits == 0:
            print("t: {:05d}, loss: {:4.4e}, loss x: {:4.4e}"
                  "".format(t, losses_y[-1], loss_x.sum()), end="\n")
            x_im = prior.x_hat.detach().clone()
            x_splits["x"].append(x_im)
            x_splits["t"].append(t)

            if display:
                ax = plt.subplot(rows, cols, int(t/niter * verbose_splits) + 1)
                ax.set_title('t = ' + str(t))
                if len(prior.x_hat.shape) == 3:
                    im = prior.x_hat[0, :, :]
                else:
                    im = prior.x_hat[0, 0, :, :]
                ax.imshow(im.detach().cpu().numpy())

        # keep track of best reconstruction
        best_resd.data = torch.min(best_resd.cpu().detach(), resds_y[-1].cpu())
        idx_improve = best_resd == resds_y[-1].cpu() 
        x_best.data[idx_improve, ...] = prior.x_hat.detach()[idx_improve, ...]

        if idx_improve.sum() > 0 and 'return_model' in args_prior.keys() and args_prior['return_model']:
            best_dd = (copy.deepcopy(prior.model), copy.deepcopy(prior.z_hat))

    if display:
        plt.tight_layout()
        plt.show()

    if 'return_model' in args_prior.keys() and args_prior['return_model']:
        return np.array(losses_y), torch.stack(resds_y).cpu().numpy(), np.array(losses_x), x_splits, x_best, best_dd
    else:
        return np.array(losses_y), torch.stack(resds_y).cpu().numpy(), np.array(losses_x), x_splits, x_best

