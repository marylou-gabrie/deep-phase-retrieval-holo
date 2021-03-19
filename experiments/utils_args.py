import ast
import numpy as np
import os
import torch

def str_to_tup(str_):
    tup_ = ast.literal_eval(str_)

    if type(tup_) is str:
        tup_ = ast.literal_eval(tup_)

    return tup_

def tup_to_dic(noise_tup=(None,),
               ref_tup=('binary', 0), 
               prior_tup=(None,),
               opt_tup=('Adam', 0.01),
               beam_tup=(None,),
               HIO_only=False):

    if noise_tup[0] is None:
        noise_dic = {'noise_type': None}
    elif noise_tup[0] == 'gaussian':
        noise_sig = noise_tup[1]
        noise_dic = {'noise_type': 'gaussian', 'noise_sig': noise_sig}
    elif noise_tup[0] == 'poisson': 
        noise_n = noise_tup[1]
        noise_dic = {'noise_type': 'poisson', 'noise_n': noise_n}

    if ref_tup[0] is None:
        ref_dic = {'method': None}
    elif 'block' in ref_tup[0]:
        if ref_tup[2] % 1 == 0:
            ref_dic = {'method': ref_tup[0], 'pad': ref_tup[1], 'pos': ref_tup[2]}
        else:
            ref_dic = {'method': ref_tup[0], 'pad': ref_tup[1], 'pos_rltv': ref_tup[2]}
        
        size_rltv = eval(ref_tup[0].split('block')[0])
        ref_dic['size_rltv'] = size_rltv
    else:
        ref_dic = {'method': ref_tup[0], 'pad': ref_tup[1]}

    if beam_tup[0] is None:
        beamstop_dic = {'beamstop_mask': None}
    elif beam_tup[0] == 'size':
        beamstop_dic = {'beamstop_mask': beam_tup[1]}

    if HIO_only:
        return noise_dic, ref_dic, beamstop_dic

    if prior_tup[0] is None:
        prior_dic = {'method': None}
        if len(prior_tup) > 1:
            prior_dic['init_mode'] = prior_tup[1]
    elif prior_tup[0] == 'deepdecoder':
        prior_dic = {'method': 'deepdecoder'}
        if len(prior_tup) > 1:
            prior_dic['depth'] = prior_tup[1]
        if len(prior_tup) > 2:
            prior_dic['channels'] = prior_tup[2]
        if len(prior_tup) > 3:
            if prior_tup[3] == 'nosig':
                prior_dic['need_sigmoid'] = False
            elif prior_tup[3] == 'sig':
                prior_dic['need_sigmoid'] = True

    opt_dic = {'method': opt_tup[0], 'lr': opt_tup[1]}

    return noise_dic, opt_dic, prior_dic, ref_dic, beamstop_dic

def process_args_to_dic(args, HIO_only=False):
    noise_tup = str_to_tup(args.args_noise)
    ref_tup = str_to_tup(args.args_ref)

    ## FIXMEME 
    ## I had added the except to avoid an error with my code.
    ## Could use a recheck
    try:
        beam_tup = str_to_tup(args.args_beamstop)
    except:
        beam_tup = (None,)

    if not HIO_only:
        opt_tup = str_to_tup(args.args_opt)
        prior_tup = str_to_tup(args.args_prior)

        args.args_noise, args.args_opt, args.args_prior, args.args_ref, args.args_beamstop = tup_to_dic(
                noise_tup=noise_tup,
                prior_tup=prior_tup,
                opt_tup=opt_tup,
                ref_tup=ref_tup,
                beam_tup=beam_tup)

        return args
    else:
        args.args_noise, args.args_ref, args.args_beamstop = tup_to_dic(
                noise_tup=noise_tup,
                ref_tup=ref_tup,
                beam_tup=beam_tup,
                HIO_only=HIO_only)
        return args

