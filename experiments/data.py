"""
Script to load data images for experiments
"""
import numpy as np
import os
import skimage
import torch
from torchvision import datasets, transforms
from PIL import Image
from scipy.stats import mode

# PHASE_RETRIEVAL_EXPERIMENTS_DATADIR = os.environ.get('PHASE_RETRIEVAL_EXPERIMENTS_DATADIR', None)
# if PHASE_RETRIEVAL_EXPERIMENTS_DATADIR is not None:
#     home = PHASE_RETRIEVAL_EXPERIMENTS_DATADIR
# else:
#     username = pwd.getpwuid(os.getuid()).pw_name
    
#     ceph_dir = os.path.join("/mnt", "ceph", "users", username)
#     if os.path.isdir(ceph_dir):
#         home = ceph_dir
#     elif os.path.isdir('/mnt/ceph/users/mgabrie/'):
#         home = '/mnt/ceph/users/mgabrie/'
#     elif os.path.isdir('/Users/mgabrie/Dropbox/Postdoc/Experiments/ceph/'):
#         home = '/Users/mgabrie/Dropbox/Postdoc/Experiments/ceph/'
#     else:
#         raise RuntimeError('Data path not understood')

# ceph_home = os.path.join(home, 'deep-phase-retrieval/')
# data_home = os.path.join(home, 'data/')

def np_relu(x, in_place=True):
    if not in_place:
        x = x.clone()
    x[x < 0] = 0
    return x

def zero_background(x):
    old_shape = x.shape
    shape = list(old_shape[:-1])
    shape[-1] = x.shape[-1] * x.shape[-2]
    x = x.reshape(shape)
    mode_ = mode(x, axis=0)[0][0]
    assert mode_ < 60
    x = np_relu(x - mode_)
    x = x.reshape(old_shape)
    
    return x

def load_data(args_data, n_image=1, device='cpu', data_home=''):
    if args_data == 'MNIST':
        train_data = datasets.MNIST(data_home, train=True, download=True,
                                    transform=transforms.ToTensor())
        x = train_data.data[0:n_image].reshape(n_image, 28, 28).double()
        x = x.to(device) / 255

    elif args_data == 'VIRUS':
        if n_image > 1:
            raise RuntimeError('Single image available for VIRUS')
        data = Image.open(data_home + '/VIRUS/mimivirus.png').convert('L')
        dim = 64
        data = data.resize((dim, dim), Image.ANTIALIAS)
        x = (torch.tensor(np.array(data)).double().to(device) / 255)
        x = x.reshape(1, dim, dim)

    elif args_data == 'VIRUS256':
        if n_image > 1:
            raise RuntimeError('Single image available for VIRUS256')
        data = Image.open(data_home + '/VIRUS/mimivirus.png').convert('L')
        dim = 256
        data = data.resize((dim, dim), Image.ANTIALIAS)
        x = (torch.tensor(np.array(data)).double().to(device) / 255)
        x = x.reshape(1, dim, dim)

    elif args_data == 'CAMERA':
        if n_image > 1:
            raise RuntimeError('Single image available for CAMERA')
        data = skimage.data.camera()
        data = Image.fromarray(data)
        dim = 128
        data = data.resize((dim, dim), Image.ANTIALIAS)
        x = (torch.tensor(np.array(data)).double().to(device) / 255)
        x = x.reshape(1, dim, dim)

    elif args_data == 'COIL':
        if n_image > 1:
            raise RuntimeError('Single image available from COIL in demo repo')
        xs = []
        for n in range(n_image):
            # obj_num = np.random.randint(100) + 1
            # rot_num = np.random.randint(int(360/5)) * 5
            obj_num = n + 1
            rot_num = 0
            coil_loc = data_home 
            coil_loc += '/COIL100/obj{}__{}.png'.format(obj_num, rot_num)
            x = np.array(Image.open(coil_loc))
            x = np.mean(x, -1)
            x = zero_background(x)
            x = torch.tensor(x).double().to(device) / 255
            x = x.unsqueeze(0)
            xs.append(x)

        x = torch.cat(xs, dim=0)
    return x