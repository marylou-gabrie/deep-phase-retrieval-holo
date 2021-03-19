import numpy as np
import os
import skimage
import torch
import matplotlib.pyplot as plt
from phase_retrieval_code_demo.core.forward_model import Obs
from phase_retrieval_code_demo.core.optimize import fit
from PIL import Image

torch.manual_seed(10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device, ', device)

data_home = '../data/'

# args_noise = {'noise_type': 'poisson', 'noise_n': 1e3}
# args_noise = {'noise_type': 'gaussian', 'noise_sig': 1e-3}
args_noise = {'noise_type': None}

args_opt = {'method': 'Adam', 'lr': 0.05}
# args_opt={'method': 'LBFGS', 'lr': 0.01}
niter = 100

# args_prior = {'method': 'deepdecoder', 'depth': 2, 'channels': 64,
#               'need_sigmoid': False}
args_prior = {'method': None, 'init_mode': 'rand'}

args_ref = {'method': 'binary', 'pad': 1}
# args_ref = {'method': None}

# loss_type = 'squared'
loss_type = 'poisson'
run_parallel = 3

oversampling = 3
comp = False

### Signal - CAMERA from skimage
data = skimage.data.camera()
data = Image.fromarray(data)
dim = 128
data = data.resize((dim, dim), Image.ANTIALIAS)
x = (torch.tensor(np.array(data)).double().to(device) / 255)
x = x.reshape(1, dim, dim)
x = x.repeat_interleave(run_parallel, 0) 

nim, nx, ny = x.shape
if comp:
    x_complex = torch.zeros(nim, 2, nx, ny)
    x_complex[:, 0, :, :] = x
    x = x_complex


### Observations
obs = Obs(args_ref=args_ref, args_noise=args_noise,
          oversampling=oversampling, device=device)

x_comp = obs.pad_and_ref(x) 
y = obs.forward(x_comp) 


## Reconstruction
_ = fit(y, obs, niter=niter,
    args_opt=args_opt,
    args_prior=args_prior,
    loss_type=loss_type, 
    x_true=x, 
    verbose_splits=5,
    display=display,
    device=device)

if 'return_model' in args_prior.keys() and args_prior['return_model'] == True:
    losses_y, losses_x, x_splits, x_best, best_dd = _
else:
    losses_y, resds_y, losses_x, x_splits, x_best = _ 

plt.figure()
for run in range(resds_y.shape[1]):
    plt.plot(resds_y[:, run])
plt.xlabel('iterations')
plt.ylabel('resds_y')