import torch
import numpy as np
import time

def name_file(args, date=False, opt_args=True):    
    filename = args.data + '/'

    if date: 
        filename += time.strftime('%d-%m-%Y') + '_'

    if args.args_ref['method'] is not None:
        filename += 'holographic'
        filename += '_' + 'pad' + str(args.args_ref['pad'])
        filename += '_' + args.args_ref['method']
        if args.args_ref['method'] == 'pinholeml' or 'block' in  args.args_ref['method']:
            try:
                filename += str(args.args_ref['pos'])
            except:
                filename += str(args.args_ref['pos_rltv'])
    else:
        filename += 'classic'

    if 'over_sampling' in args.__dict__.keys():
        filename += '_ovs{:0.1f}'.format(args.over_sampling)
    else:
        filename += '_ovs2.0'
    

    if args.args_noise['noise_type'] is None:
        filename += '_noise0'
    else:
        filename += '_noise' + args.args_noise['noise_type']
        if args.args_noise['noise_type'] == 'gaussian':
            filename += str(args.args_noise['noise_sig'])
        elif args.args_noise['noise_type'] == 'poisson':
            filename += str(args.args_noise['noise_n'])
    
    if args.beamstop_area_fraction is not None:
        shape, area = args.beamstop_area_fraction.split(",")
        area = float(area)
        filename += f"__{shape}_beamstop_area_{area:1.5f}_"
    
    if args.baseline is not None:
        filename += f"__baseline_{args.baseline.replace(',', '_').replace('.', '_point_')}_"
    elif opt_args is True:
        if args.args_prior['method'] is None:
            filename += '_priorno'
            if 'init_mode' in args.args_prior.keys():
                filename += args.args_prior['init_mode']
        else:
            filename += '_prior' + args.args_prior['method']

            if args.args_prior['method'] == 'deepdecoder':
                if 'need_sigmoid' in args.args_prior.keys():
                    if not args.args_prior['need_sigmoid']:
                        filename += 'nosig'
                if 'depth' in args.args_prior.keys():
                    filename += str(args.args_prior['depth'])
                else:
                    filename += '2'

                if 'channels' in args.args_prior.keys():
                    filename += 'c' + str(args.args_prior['channels'])

        if args.complex == 1:
            filename += '_comp'
        
        filename += '_loss' + args.loss_type\
                + '_' + args.args_opt['method'] + str(args.args_opt['lr'])

    if args.args_beamstop['beamstop_mask'] is None:
        filename += '_beam0'
    else:
        filename += '_beam' + str(args.args_beamstop['beamstop_mask'])

    return filename
