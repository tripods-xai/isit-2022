__author__ = 'yihanjiang, abhmul'
'''
Evaluate
'''
from src.utils import get_test_sigmas, new_seed, binlen
from src.turbo_benchmarker import TurboBenchmarker

import numpy as np
from numpy.random import default_rng

import argparse

from commpy import channelcoding as cc
from commpy.utilities import dec2bitarray

from tqdm.contrib.concurrent import process_map
from tqdm import trange

parser = argparse.ArgumentParser()

parser.add_argument('--num_block', type=int, default=100)
parser.add_argument('--block_len', type=int, default=100)
parser.add_argument('--num_dec_iteration', type=int, default=6)

parser.add_argument('--g1',  type=int, default=7)  # ,7,5,7 is the best
parser.add_argument('--g2',  type=int, default=5)
parser.add_argument('--g3',  type=int, default=5)
parser.add_argument('--feedback',  type=int, default=7)
parser.add_argument('--systematic', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--decoder', choices=['modified_hazzys', 'modified_basic', 'tensorflow_basic'], default='modfied_basic')

parser.add_argument('--num_cpu', type=int, default=1)

parser.add_argument('--snr_test_start', type=float, default=-1.5)
parser.add_argument('--snr_test_end', type=float, default=2.0)
parser.add_argument('--snr_points', type=int, default=8)

parser.add_argument('--save', action='store_true',
                    help="Save results to file when complete")
parser.add_argument('--plot', action='store_true',
                    help="Create a plot when complete")
parser.add_argument('--show_plot', action='store_true',
                    help="Show plot when complete")

SEED = 2021

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model_title = f'turbo-{args.g1}{args.g2}{args.g3}-{args.feedback}'

    memory = max(binlen(g) for g in [args.g1, args.g2, args.g3]) - 1
    # Number of delay elements in the convolutional encoder
    M = np.array([memory])
    # Encoder of convolutional encoder
    generator_matrix1 = np.array([[args.g1, args.g2]])
    # Encoder of convolutional encoder
    generator_matrix2 = np.array([[args.g1, args.g3]])
    feedback = args.feedback  # Feedback of convolutional encoder

    print('[testing] Turbo Code Encoder 1: G: ',generator_matrix1, 'Feedback: ', feedback,  'M: ', M)
    print(f'[testing] Generator 1: {dec2bitarray(generator_matrix1[0, 0], memory + 1)}')
    print(f'[testing] Generator 2: {dec2bitarray(generator_matrix1[0, 1], memory + 1)}')
    trellis1 = cc.Trellis(M, generator_matrix1, feedback=feedback)
    trellis1.delay = 0
    print('[testing] Turbo Code Encoder 2: G: ',generator_matrix2, 'Feedback: ', feedback,  'M: ', M)
    print(f'[testing] Generator 1: {dec2bitarray(generator_matrix2[0, 0], memory + 1)}')
    print(f'[testing] Generator 3: {dec2bitarray(generator_matrix2[0, 1], memory + 1)}')
    trellis2 = cc.Trellis(M, generator_matrix2, feedback=feedback)
    trellis2.delay = 0

    snrs, test_sigmas = get_test_sigmas(
        args.snr_test_start, args.snr_test_end, args.snr_points)

    benchmarker = TurboBenchmarker(
        trellis1,
        trellis2,
        systematic=args.systematic,
        model_title=model_title,
        save_to_file=args.save,
        save_plot=args.plot,
        show_plot=args.show_plot, seed_generator=default_rng(SEED))
    benchmarker.benchmark(
        encoder_name="modified_basic",
        decoder_name=args.decoder,
        test_sigmas=test_sigmas,
        num_block=args.num_block,
        block_len=args.block_len,
        num_decoder_iterations=args.num_dec_iteration,
        num_cpu=args.num_cpu,
        batch_size=args.batch_size
    )
