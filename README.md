# HOLOGRAPHIC PHASE RETRIEVAL DEMO
 
The following repository contains a package 'phase_retrieval_code_demo' and scripts to replicate experiments of "Phase Retrieval with Holography and Untrained Priors: Tackling the Challenges of Low-Photon Nanoscale Imaging" Hannah Lawrence, David A. Barmherzig, Henry Li, Michael Eickenberg, Marylou Gabrié. The general package can be used for future research on phase retrieval.
A demo notebook for usage of the package is also made available.
[arxiv:2012.07386](https://arxiv.org/abs/2012.07386)

## Requirements
The package has been tested with

- matplotlib==3.1.0
- numpy==1.16.4
- Pillow==6.0.0
- scikit-image==0.17.2
- scipy==1.4.1
- torch==1.4.0
- torchvision==0.5.0

## Package installation
from the root of this repository run:

```
python setup.py install --user
```

## How to pass arguments to the package and experiment files ?
The package in itself accepts some dictionaries to regroup arguments corresponding to different conceptual blocks. Dictionaries makes things clear thanks to the keys. However, in order to be able to run experiments from command lines, for the experiments like "run_reconstruction_holo_opt.py" the dictionaries have tuples equivalent. Here we describe the different possible combination of arguments, how to pass them as dictionaries to the functions and classes of the packages and finally what are their tuples equivalent for the experiment files.

### 1/ For observations: args_noise, args_ref and args_beamstop

#### 1.1/ args_noise a.k.a --arg-noise or -an


For a noiseless simulation
```
args_noise = {'noise_type': None}
-an (None,)
```
For Poisson noise as in the paper:
noise_n corresponds to N_p, the number of photons/pixel
```
args_noise = {'noise_type': 'poisson', 'noise_n': 1e3}
-an ('poisson',1e3)
```
For additive Gaussian noise
noise_sig corresponds to variance of the noise
```
args_noise = {'noise_type': 'gaussian', 'noise_sig': 1e-3}
-an ('gaussian',1e-3)
```

#### 1.1/ args_ref a.k.a. --args-ref or -arf


For a simulation without reference - classical phase retrieval
```
args_ref = {'method': None}
-arf (None,)
```
For a simulation with a reference, 
- 'pad' (int) corresponds to the position of the reference 
    with respect to the signal, ref separated by as many blocks 
    of zeros of the size of the signal
- 'method' correspond to the values
```
# binary random 0-1 entries, right next to the signal
args_ref = {'method': 'binary', 'pad': 0}
-arf ('binary',0)
# random uniform 0-1 entries, 
# separated by a block of zeros of the size of the signal 
args_ref = {'method': 'rand', 'pad': 1}
-arf ('rand',1)
# custom ref, where x_ref as same size as signal
args_ref = {'method': 'custom', 'x_ref'=x_ref}
## no-command line translation
```

For experiments small block reference is passed as custom
to packages like the last line above. The experiment files 
create this custom ref from a tuple:
- prefix of first element fixes relative size (below 0.1)
- second element remains separation as above
- third argument gives relative position of small block ref in the frame after separation as been taken into account.
```
-arf ('0.1blockbinary',0,0.5)
```

#### 1.3/ args_beamstop 
#TODO

### 2/ For optimization: args_opt, args_prior

#### 2.1/ args_opt a.k.a. --args-opt or -ao


Fixing the optimizer and learning rate,
supports SGD, Adam and LBFGS
```
args_opt = {'method': 'Adam', 'lr': 0.05}
-ao ('Adam',0.05)
args_opt={'method': 'LBFGS', 'lr': 0.01}
-ao ('LBFGS',0.01)
args_opt={'method': 'SGD', 'lr': 0.1}
-ao ('SGD',0.1)
``` 

#### 2.2/ args_prior a.k.a. --args-prior or -ap


For optimization directly on the pixels,
Optionally inititalization can be chosen:
- fixing 'init_mode' 
    -random uniform between 0-1 'rand'
    -random normal with variance 1 'randn' (default)
- giving 'x_init'
```
args_prior = {'method': None, 'init_mode': 'rand'}
-ap (None,'rand')
args_prior = {'method': None, 'init_mode': 'randn'}
-ap (None,'randn')
args_prior = {'method': None, 'x_init': 'rand'}
## no command line equivalent
```
For optimization with a deep decoder,
Optionally:
    - depth and channels can be specified, (default 2-64)
    - need_sigmoid (default True)
```
args_prior = {'method': 'deepdecoder'}
-ap ('deepdecoder')
args_prior = {'method': 'deepdecoder', 'depth': 2}
-ap ('deepdecoder',2)
args_prior = {'method': 'deepdecoder', 'depth': 2, 'channels': 64}
-ap ('deepdecoder',2,62)
args_prior = {'method': 'deepdecoder', 'depth': 2, 'channels': 64, 'need_sigmoid': False}
-ap ('deepdecoder',2,62,'nosig')
args_prior = {'method': 'deepdecoder', 'depth': 2, 'channels': 64, 'need_sigmoid': True}
-ap ('deepdecoder',2,62,'sig')