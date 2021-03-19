import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class NoPrior(nn.Module):
    """
    Class to optimize directly over the pixels
    """
    def __init__(self, size_output, x_init=None,
                init_mode='randn', 
                 device='cpu', domain='real'):
        if x_init is not None:
            self.x_hat = x_init.double().to(device)
        else:
            print("Pixel initialization with mode: ", init_mode)
            if init_mode == 'randn':
                self.x_hat = torch.randn(size_output).double().to(device)
            elif init_mode == 'rand':
                self.x_hat = torch.rand(size_output).double().to(device)
            else:
                raise RuntimeError("Init mode not understood")
        self.x_hat.requires_grad_()


"""
Code adapted from ... FIXMEML
"""

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

def conv(in_f, out_f, kernel_size, stride=1, pad='zero', groups=1):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, 
                          padding=to_pad, bias=False, groups=groups)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


def decodernw(
        num_output_channels=3,
        num_channels_up=[128]*5,
        num_parallel=1,
        filter_size_up=1,
        need_sigmoid=True,
        pad='reflection',
        upsample_mode='bilinear',
        act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
        bn_before_act=False,
        bn_affine=True,
        upsample_first=True,
        ):

    num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
    num_channels_up = [n * num_parallel for n in num_channels_up]
    n_scales = len(num_channels_up)

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up]*n_scales
    
    
    model = nn.Sequential()
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add(conv(num_channels_up[i], num_channels_up[i+1], filter_size_up[i], 1, 
                      pad=pad, groups=num_parallel))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
            model.add(conv(num_channels_up[i], num_channels_up[i+1], filter_size_up[i], 1,
                           pad=pad, groups=num_parallel))        
        
        if i != len(num_channels_up) - 1:	
            if(bn_before_act): 
                model.add(nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))
            model.add(act_fun)
            if(not bn_before_act): 
                model.add(nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad, groups=num_parallel))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

class DDPrior():
    """
    Class to optimize using a deep decoder prior.
    The class is able to optimize for several images in parallel.
    The parallelization is done channel-wise,
    with no interactions between channels.
    The size of the latent variable is automatically computed from
    depth and size_output informations

    Parameters
    ===========
    num_channels_up: lists
        each number of channels for each layer
        length determines the depth

    size_output: tuple
        - (nim, nx, ny) (real)
        - (nim, 2, nx, ny) (complex)

    init: either None or deep decoder model
        allows to resume from a deep decoder model that was previously saved

    need_sigmoid: bool
        True adds a sigmoid activation at last layer
    """
    def __init__(self, size_output, num_channels_up=[50]*2, device='cpu',
                 need_sigmoid=True):
        self.device = device

        if len(size_output) == 4:  # complex case
            num_output_channels = 2 * size_output[0]
            self.domain = 'complex' 
        else:
            num_output_channels = size_output[0] 
            self.domain = 'real'
        
        totalupsample = 2 ** len(num_channels_up)

        if (size_output[-2] % totalupsample) != 0 or (size_output[-1] % totalupsample) != 0:
            raise ValueError("the image output of a deep decoder must be"
                             "divisible by 2 ** (number of layers)")

        width_z = int(size_output[-2] / totalupsample)
        height_z = int(size_output[-1] / totalupsample)

        shape = [1, size_output[0] * num_channels_up[0], width_z, height_z]

        self.z_hat = Variable(torch.zeros(shape)).to(device)

        self.model = decodernw(num_output_channels=num_output_channels,
                               num_channels_up=num_channels_up,
                               num_parallel=size_output[0],
                               need_sigmoid=need_sigmoid).to(device)

        self.z_hat.data.uniform_()
        self.z_hat.data *= 1./10
        
        self.x_hat = torch.zeros(size_output).to(device)
        self.params = [p for p in self.model.parameters()]
        x_hat = self.model(self.z_hat)        

    def decode(self, z):
        """
        Function returning output x image from latent variable z
        """
        # getting rid of the pytorch batch-dim channel dimension
        # in our case, the channel is used as the batch dimension to treat several images in parallel
        if self.domain == 'real':
            return self.model(z)[0, :, :, :]
        else:
            x = self.model(z)[0, :, :, :]
            return x.reshape(int(x.shape[0]/2), 2, x.shape[-2], x.shape[-1])