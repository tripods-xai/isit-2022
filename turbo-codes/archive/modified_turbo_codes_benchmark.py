__author__ = 'abhmul'
'''
Evaluate
'''
import numpy as np
from numpy.random import default_rng

import argparse

from commpy.utilities import dec2bitarray

from src.utils import get_test_sigmas, binlen
from src.turbo_benchmarker import TurboBenchmarker
import src.modified_convcode as mcc

parser = argparse.ArgumentParser()

parser.add_argument('--num_block', type=int, default=100)
parser.add_argument('--block_len', type=int, default=100)
parser.add_argument('--num_dec_iteration', type=int, default=6)

parser.add_argument('--g1',  type=int, default=31)     #,7,5,7 is the best
parser.add_argument('--g2',  type=int, default=23)
parser.add_argument('--g3',  type=int, default=30)

parser.add_argument('--b1',  type=int, default=1)     #,7,5,7 is the best
parser.add_argument('--b2',  type=int, default=0)
parser.add_argument('--b3',  type=int, default=1)

parser.add_argument('--rsc', action='store_true', help="Run the code as a recursive systematic code instead")

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--decoder', choices=['modified_hazzys', 'modified_basic', 'tensorflow_basic'], default='modfied_basic')
parser.add_argument('--num_cpu', type=int, default=1)

parser.add_argument('--snr_test_start', type=float, default=-1.5)
parser.add_argument('--snr_test_end', type=float, default=2.0)
parser.add_argument('--snr_points', type=int, default=8)

parser.add_argument('--save', action='store_true', help="Save results to file when complete")
parser.add_argument('--plot', action='store_true', help="Create a plot when complete")
parser.add_argument('--show_plot', action='store_true', help="Show plot when complete")

SEED = 2021

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    memory = max(binlen(g) for g in [args.g1, args.g2, args.g3]) - 1
    gen1 = dec2bitarray(args.g1, memory + 1)
    bias1 = args.b1
    gen2 = dec2bitarray(args.g2, memory + 1)
    bias2 = args.b2
    gen3 = dec2bitarray(args.g3, memory + 1)
    bias3 = args.b3

    model_title = f'modified-turbo-{"rsc-" if args.rsc else ""}{args.g1}{args.g2}{args.g3}-{args.b1}{args.b2}{args.b3}'
    
    trellis1 = mcc.ModifiedTrellis(np.stack([gen1, gen2], axis=0), np.stack([bias1, bias2], axis=0), delay=0)
    trellis2 = mcc.ModifiedTrellis(np.stack([gen1, gen3], axis=0), np.stack([bias1, bias3], axis=0), delay=0)
    if args.rsc:
        trellis1 = trellis1.to_rsc()
        trellis2 = trellis2.to_rsc()
    
    print("trellis1")
    print(f"Trellis is code type {trellis1.code_type} and rate {trellis1.k}/{trellis1.n}")
    print(f"Trelis has generator {trellis1.generator} and bias {trellis1.bias}")
    print(f"Code has delay {trellis1.delay}")
    print("Output table")
    print(trellis1.output_table)
    print("Next State table")
    print(trellis1.next_state_table)

    print("trellis2")
    print(f"Trellis is code type {trellis2.code_type} and rate {trellis2.k}/{trellis2.n}")
    print(f"Trelis has generator {trellis2.generator} and bias {trellis2.bias}")
    print(f"Code has delay {trellis2.delay}")
    print("Output table")
    print(trellis2.output_table)
    print("Next State table")
    print(trellis2.next_state_table)


    snrs, test_sigmas = get_test_sigmas(
        args.snr_test_start, args.snr_test_end, args.snr_points)

    benchmarker = TurboBenchmarker(
        trellis1, 
        trellis2, 
        systematic=args.rsc,
        model_title=model_title, 
        save_to_file=args.save, 
        save_plot=args.plot, 
        show_plot=args.show_plot, 
        seed_generator=default_rng(SEED))
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

