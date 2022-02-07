__author__ = 'yihanjiang, abhmul'
'''
Evaluate
'''
from utils import  awgn_corrupt, get_test_sigmas, new_seed

import numpy as np
from numpy.random import default_rng

import time
from datetime import datetime
import sys
import os
from pathlib import Path

from commpy import channelcoding as cc
from commpy.channelcoding import turbo
from commpy.utilities import hamming_dist, dec2bitarray

import tests.modified_turbo as mt
import tests.modified_convcode as mcc

from tqdm.contrib.concurrent import process_map
from tqdm import trange

import plotly.express as px

SEED = 2021
rng = default_rng(SEED)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_block', type=int, default=100)
    parser.add_argument('--block_len', type=int, default=100)
    parser.add_argument('--num_dec_iteration', type=int, default=6)

    parser.add_argument('--code_rate',  type=int, default=3)

    parser.add_argument('--M',  type=int, default=2)
    parser.add_argument('--enc1',  type=int, default=7)     #,7,5,7 is the best
    parser.add_argument('--enc2',  type=int, default=5)
    parser.add_argument('--enc3',  type=int, default=5)
    parser.add_argument('--feedback',  type=int, default=7)

    parser.add_argument('--num_cpu', type=int, default=1)

    parser.add_argument('--snr_test_start', type=float, default=-1.5)
    parser.add_argument('--snr_test_end', type=float, default=2.0)
    parser.add_argument('--snr_points', type=int, default=8)

    parser.add_argument('--id', type=str, default=str(rng.random())[2:8])
    
    parser.add_argument('--plot', action='store_true', help="Create a plot when complete")
    parser.add_argument('--show_plot', action='store_true', help="Show plot when complete")
    parser.add_argument('--debug', action='store_true', help='Runs in debug mode w/o multiprocessing')

    args = parser.parse_args()
    print(args)
    print('[ID]', args.id)
    return args

if __name__ == '__main__':
    args = get_args()

    if args.debug:
        print('Running in DEBUG mode')

    M = np.array([args.M])                                  # Number of delay elements in the convolutional encoder
    generator_matrix1 = np.array([[args.enc1, args.enc2]])   # Encoder of convolutional encoder
    generator_matrix2 = np.array([[args.enc1, args.enc3]])   # Encoder of convolutional encoder
    feedback = args.feedback                 # Feedback of convolutional encoder
    
    print('[testing] Turbo Code Encoder 1: G: ', generator_matrix1,'Feedback: ', feedback,  'M: ', M)
    print(f'[testing] Generator 1: {dec2bitarray(generator_matrix1[0, 0], args.M + 1)}')
    print(f'[testing] Generator 2: {dec2bitarray(generator_matrix1[0, 1], args.M + 1)}')
    trellis1 = cc.Trellis(M, generator_matrix1 , feedback=feedback)
    trellis1.delay = 0
    print('[testing] Turbo Code Encoder 2: G: ', generator_matrix2,'Feedback: ', feedback,  'M: ', M)
    print(f'[testing] Generator 1: {dec2bitarray(generator_matrix2[0, 0], args.M + 1)}')
    print(f'[testing] Generator 3: {dec2bitarray(generator_matrix2[0, 1], args.M + 1)}')
    trellis2 = cc.Trellis(M, generator_matrix2, feedback=feedback)
    trellis2.delay = 0
    interleaver = cc.RandInterlv(args.block_len, new_seed(rng))

    snrs, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    turbo_res_ber, turbo_res_bler= [], []

    tic = time.time()

    def turbo_compute(run_args):
        '''
        Compute Turbo Decoding in 1 iterations for one SNR point.
        '''
        (idx, x) = run_args
        message_bits = rng.integers(0, 2, args.block_len)
        # print(hash(tuple(message_bits)))
        [sys, par1, par2] = mt.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        sys_r  = awgn_corrupt(rng, sys, sigma = test_sigmas[idx])
        par1_r = awgn_corrupt(rng, par1, sigma = test_sigmas[idx])
        par2_r = awgn_corrupt(rng, par2, sigma = test_sigmas[idx])

        # decoded_bits = turbo.turbo_decode(
        #     sys_r, par1_r, par2_r, 
        #     trellis1,
        #     test_sigmas[idx]**2, 
        #     args.num_dec_iteration, 
        #     interleaver
        # )

        decoded_bits = mt.turbo_decode(
            sys_r, par1_r, par2_r, 
            trellis1, trellis2,
            test_sigmas[idx]**2, 
            args.num_dec_iteration, 
            interleaver
        )

        # decoded_bits = mt.hazzys_turbo_decode(
        #     sys_r, par1_r, par2_r, 
        #     trellis1, trellis2,
        #     test_sigmas[idx]**2, 
        #     args.num_dec_iteration, 
        #     interleaver
        # )

        num_bit_errors = hamming_dist(message_bits, decoded_bits)

        return num_bit_errors


    commpy_res_ber = []
    commpy_res_bler= []

    nb_errors = np.zeros(test_sigmas.shape)
    map_nb_errors = np.zeros(test_sigmas.shape)
    nb_block_no_errors = np.zeros(test_sigmas.shape)

    for idx in range(len(test_sigmas)):
        start_time = time.time()
        if args.num_cpu == 1:
            results = [turbo_compute((idx, x)) for x in trange(int(args.num_block))]
        else:
            results = process_map(turbo_compute, [(idx,x) for x in range(int(args.num_block))], max_workers=args.num_cpu, chunksize=args.num_block // 100)

        for result in results:
            if result == 0:
                nb_block_no_errors[idx] = nb_block_no_errors[idx]+1

        nb_errors[idx]+= sum(results)
        print('[testing]SNR: ' , snrs[idx])
        print('[testing]BER: ', sum(results)/float(args.block_len*args.num_block))
        print('[testing]BLER: ', 1.0 - nb_block_no_errors[idx]/args.num_block)
        commpy_res_ber.append(sum(results)/float(args.block_len*args.num_block))
        commpy_res_bler.append(1.0 - nb_block_no_errors[idx]/args.num_block)
        end_time = time.time()
        print('[testing] This SNR runnig time is', str(end_time-start_time))


    print('[Result]SNR: ', snrs)
    print('[Result]BER', commpy_res_ber)
    print('[Result]BLER', commpy_res_bler)

    toc = time.time()
    print('[Result]Total Running time:', toc-tic)

    if args.plot:
        model_title = f'Turbo-{args.enc1}{args.enc2}{args.feedback}'
        fig = px.line(
                x=snrs, y=commpy_res_ber, 
                title=f'{model_title} BER vs. SNR', 
                labels={'x': 'SNR', 'y': 'BER'}, log_y=True
            )
        
        time_format = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        fig.write_image(f'../images/{model_title}.{time_format}.png')
        if args.show_plot:
            fig.show()
