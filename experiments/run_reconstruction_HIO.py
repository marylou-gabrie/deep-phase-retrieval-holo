import argparse
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

from phase_retrieval_code_demo.core.forward_model import Obs
from phase_retrieval_code_demo.baselines.HIO import HIO
from phase_retrieval_code_demo.core.utils import cosine_distance_invariant, get_beamstop_masks

from data import load_data
from utils_args import process_args_to_dic
from utils_io import name_file

# Determine if gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get arguments
## CAREFUL: There should not be any space in the string passing the tuples 
## as command lines arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-an', '--args-noise', type=str, default='(None,)')
parser.add_argument('-arf', '--args-ref', type=str, default='("binary",0)')
parser.add_argument('-ovs', '--over-sampling', type=float, default=2)
parser.add_argument('-comp', '--complex', type=int, default=0)
parser.add_argument('-postv', '--positive', type=int, default=1)
parser.add_argument('-it', '--n-iter', type=int, default=10)
parser.add_argument('-rp', '--run-parallel', type=int, default=1)
parser.add_argument('-ni', '--n-images', type=int, default=1)
parser.add_argument('-d', '--data', type=str, default='MNIST')
parser.add_argument('-rsd', '--random-seed', type=int, default=10)
parser.add_argument('-id', '--slurm-id', type=str, default='0')
parser.add_argument('--baseline', default=None)
parser.add_argument('--beamstop-area-fraction', default=None, type=float)

args = parser.parse_args()
torch.manual_seed(args.random_seed)

args = process_args_to_dic(args, HIO_only=True)

filename = '../tmp/' + name_file(args, date=True, opt_args=False) + '_hio' + '_rsd' + str(args.random_seed)
if args.slurm_id != 0:
    filename += '_id' + str(args.slurm_id)
filename += '.pickle'

print(filename)

# Load data
x = load_data(args.data, data_home='../data', device=device)
x = x.repeat_interleave(args.run_parallel, 0) 
nim, nx, ny = x.shape

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

x_comp = obs.pad_and_ref(x)
y = obs.forward(x_comp)

if args.beamstop_area_fraction is not None:
    from phase_retrieval_code_demo.hannah.utils import get_beamstop_masks
    
    beamstop_mask = get_beamstop_masks(y[0].shape[1], y[0].shape[0], centered=False,
                        approx_beamstop_area_frac=float(args.beamstop_area_fraction))
    obs = Obs(args_ref=args.args_ref, args_noise=args.args_noise, 
          oversampling=args.over_sampling, device=device,
          args_beamstop={'beamstop_mask': torch.from_numpy(beamstop_mask).to(device)})
    
    x_comp = obs.pad_and_ref(x)
    y = obs.forward(x_comp)


n_inits = 1

positive = True if args.positive == 1 else False
real = True if args.complex == 0 else False

# HIO prep for holography
if args.args_ref['method'] is not None:
    # Part 1: give it the exact reference support, in addition to the object support
    support_for_HIO = np.zeros((int(nx*args.over_sampling), int(ny*args.over_sampling)*(2 + obs.pad_ref)), dtype=np.dtype('bool'))
    support_for_HIO[0:nx, 0:ny] = True
    support_for_HIO[0:nx, (1 + obs.pad_ref) * ny:(2 + obs.pad_ref) * ny] = (abs(obs.x_ref[0, ...].cpu().numpy()) > 1e-5)
    non_support_for_HIO = np.logical_not(support_for_HIO)
    # Part 2: give it the exact reference location (via the 'mask_for_HIO' entry, which is just the appropriate block)
    #         as well as the actual values on that mask
    ref_dict = {}
    mask_for_HIO = torch.zeros((int(nx*args.over_sampling), int(ny*args.over_sampling)*(2 + obs.pad_ref)), dtype=torch.bool)
    mask_for_HIO[0:nx, (1 + obs.pad_ref) * ny:(2 + obs.pad_ref) * ny] = True
    ref_dict['ref_mask'] = mask_for_HIO
    ref_dict['ref_vals'] = obs.x_ref[0, ...].to(device).float()

    x_hio, resds_hio_y, resds_hio_x = HIO(torch.sqrt(y), real=real, 
                                           positive=positive, 
                                           n_inits = n_inits, 
                                           n_iter=args.n_iter, 
                                           non_support=non_support_for_HIO, ref_dict=ref_dict,
                                           enforce=None,
                                           beta=1., ground_truth=x_comp[:, 0, :, :], 
                                           track_errors=True)

    plt.figure(figsize=(10, 5))
    for i in range(args.run_parallel):
        plt.subplot(1,args.run_parallel,i+1)
        plt.imshow(x_hio[i, 0, 0:nx, 0:ny, 0].cpu())

    plt.show()
else:
    x_hio, resds_hio_y, resds_hio_x = HIO5(torch.sqrt(y), real=real, 
                                           positive=positive,
                                           n_inits=n_inits, n_iter=args.n_iter,
                                           beta=1., ground_truth=x_comp[:, 0, :, :],
                                           track_errors=True) 
    plt.figure(figsize=(10, 5))
    for i in range(args.run_parallel):
        plt.subplot(1,args.run_parallel,i+1)
        plt.imshow(x_hio[i, 0, 0:nx, 0:ny, 0].cpu())
    plt.show()

x_hio = x_hio.double()
x_hio_resh = x_hio[:, 0, :x.shape[-2], :x.shape[-1], 0]
loss_hio_x = (x ** 2).sum((-2, -1)) + (x_hio_resh ** 2).sum((-2, -1))
loss_hio_x = loss_hio_x - 2 * cosine_distance_invariant(x, x_hio_resh)
x_comp_hio = obs.pad_and_ref(x_hio[:, 0, :x.shape[-2], :x.shape[-1], 0]) 
y_hat_hio = obs.forward(x_comp_hio, noise_off=True)
resd_hio = torch.sqrt(((y - y_hat_hio) ** 2).sum((-2, -1))) / torch.sqrt((y ** 2).sum((-2, -1))) 

#Logs
results = {'args': args}
results['HIO_xhat'] = x_hio.detach().cpu()
results['HIO_resd'] = resd_hio
results['HIO_resds_x'] = resds_hio_x
results['HIO_resds_y'] = resds_hio_y
results['HIO_lossx'] = loss_hio_x.detach().cpu()
results['x_hio'] = x_hio
results['x_true'] = x

with open(filename, 'wb') as file:
    pickle.dump(results, file)
