import argparse
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

from phase_retrieval_code_demo.core.forward_model import Obs
from phase_retrieval_code_demo.core.optimize import fit
from phase_retrieval_code_demo.core.utils import cosine_distance_invariant, get_beamstop_masks

from data import load_data
from utils_args import process_args_to_dic
from utils_io import name_file

# Determine if gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get arguments
## CAREFUL: safer not be have space in the string passing 
## the tuples as command lines arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-an', '--args-noise', type=str, default='(None,)')
parser.add_argument('-ar', '--args-reg', type=str, default='(None,)')
parser.add_argument('-ao', '--args-opt', type=str,  default='("Adam",0.01)')
parser.add_argument('-ap', '--args-prior', type=str, default='(None,)')
parser.add_argument('-arf', '--args-ref', type=str, default='("binary",0)')
parser.add_argument('-ovs', '--over-sampling', type=float, default=2)
parser.add_argument('-comp', '--complex', type=int, default=0)
parser.add_argument('-lt', '--loss-type', type=str, default='squared')
parser.add_argument('-it', '--n-iter', type=int, default=10)
parser.add_argument('-rp', '--run-parallel', type=int, default=1)
parser.add_argument('-ni', '--n-image', type=int, default=1)
parser.add_argument('-d', '--data', type=str, default='MNIST')
parser.add_argument('-rsd', '--random-seed', type=int, default=10)
parser.add_argument('-id', '--job-id', type=str, default='0')
parser.add_argument('--baseline', default=None)
parser.add_argument('--beamstop-area-fraction', default=None, type=str)
parser.add_argument('--args-beamstop', default="(None,)")

args = parser.parse_args()
torch.manual_seed(args.random_seed)

args = process_args_to_dic(args)

filename = '../tmp/' + name_file(args,date=True) 
filename += '_holo_opt'+ '_rsd' + str(args.random_seed)
if args.job_id != 0:
    filename += '_id' + str(args.job_id)
filename += '.pickle'

# Load data
x = load_data(args.data, n_image=args.n_image,
              data_home='../data', device=device)
x = x.repeat_interleave(args.run_parallel, 0) 
nx, ny = x.shape[-2:]

# Observations
if args.args_ref['method'] is None:
    pass
elif 'blockbinary' in args.args_ref['method']:
    args.args_ref['x_ref'] = torch.zeros_like(x)
    size = int(eval(args.args_ref['method'].split('block')[0]) * nx)

    value = torch.randint(2, (size, size)).to(device) * torch.sqrt((x ** 2).sum((-2, -1))).view(x.shape[0], 1, 1) 
    value /= (0.5 * size)

    if 'pos_rltv' in args.args_ref.keys():
        pos = int(args.args_ref['pos_rltv'] * nx)
        args.args_ref['x_ref'][:, :size, pos:pos+size] = value
    else:
        pos = args.args_ref['pos']
        args.args_ref['x_ref'][:, :size, pos:pos+size] = value
    
    args.args_ref['method'] = 'custom'


obs = Obs(args_ref=args.args_ref, args_noise=args.args_noise, 
          oversampling=args.over_sampling, device=device)

if args.complex == 1: 
    ## adds complex axis to the data to allow complex reconstruction
    nim, nx, ny = x.shape
    x_complex = torch.zeros(nim, 2, nx, ny)
    x_complex[:, 0, :, :] = x
    x = x_complex

x_comp = obs.pad_and_ref(x)
y = obs.forward(x_comp)

if args.beamstop_area_fraction is not None:
    shape, area = args.beamstop_area_fraction.split(",")
    area = float(area)

    beamstop_mask = get_beamstop_masks(y[0].shape[1], y[0].shape[0],
                                       centered=False,
                                       approx_beamstop_area_frac=area,
                                       shape=shape)
    obs = Obs(args_ref=args.args_ref, args_noise=args.args_noise, 
              oversampling=args.over_sampling, device=device,
              args_beamstop={'beamstop_mask': torch.from_numpy(beamstop_mask)(device)})
    
    x_comp = obs.pad_and_ref(x)
    y = obs.forward(x_comp)


# Reconstruction
if args.baseline is None:
    losses_y, resds_y, loss_x, x_splits, x_best = fit(y, obs, niter=args.n_iter,
                                            args_opt=args.args_opt,
                                            args_prior=args.args_prior,
                                            loss_type=args.loss_type,
                                            x_true=x,
                                            verbose_splits=5,
                                            display=True,
                                            device=device)

    results = {'args': args,
            'losses_y': resds_y,
            'resds_y': resds_y,
            'loss_x': loss_x,
            'x_best': x_best,
            'x_splits': x_splits,
            'x_true': x}

    with open(filename, 'wb') as file:
        pickle.dump(results, file)

    print(filename)

elif args.baseline.split(",")[0] in ("wiener_filter", "inverse_filter"):
    # if there is a comma, then after it follows the noise level
    # if not, then use the correct noise level from the data generation
    if "," in args.baseline:
        Np = float(args.baseline.split(",")[1])
    else:
        noise = args.args_noise
        if noise['noise_type'] == None:
            Np = np.inf
        else:
            Np = float(noise['noise_n'])
    from phase_retrieval_code_demo.michael.holography_baselines import wiener_filt_torch, inv_filt_torch

    baseline_method = args.baseline.split(",")[0]

    if obs.pad_ref != 1:
        raise Exception("Padding needs to be set to 1 for wiener filtering to work")

    filename = filename.replace("_autodiff", "")

    # pick the first one of potentially multiple
    # should maybe error at multiple since that is currently not implemented
    yy = y[0]
    if len(yy.reshape(-1)) != len(y.reshape(-1)):
        raise NotImplementedError("Only works for 1 image currently")
    rr = obs.x_ref[0]
    if baseline_method == "wiener_filter":
        recovered = wiener_filt_torch(yy, rr, Np)
    elif baseline_method == "inverse_filter":
        recovered = inv_filt_torch(yy, rr)
    else:
        raise Exception("This branch is impossible")

    padded_and_reffed = obs.pad_and_ref(recovered.permute(2, 0, 1)[np.newaxis])
    y_hat = obs.forward(padded_and_reffed, noise_off=True)


    results = {'args': args,
               'x_best': recovered,
               'x_true': x,
               'y': y,
               'y_hat': y_hat,
               'ref': obs.x_ref[0],
               'beamstop': obs.beamstop_mask}
    
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

    print(filename)    
    import sys
    sys.exit(0)


