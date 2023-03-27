__author__ = 'abhmul'
'''
Evaluate
'''
from dataclasses import asdict
from pprint import pprint
from typing import cast
from src.channelcoding.encoder_decoders import TurboNonsystematicEncoderDecoder
from src.channels import CHANNELS, NoisyChannels
from src.code_trainer import ResultsWriter, Validator
from src.codes import DECODERS, ENCODERS, NonsystematicEncoders, NonsystematicDecoders
from src.dataclasses import NonsystematicTurboEncoderSpec
from src.utils import get_test_sigmas

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('encoder', type=NonsystematicEncoders, choices=list(NonsystematicEncoders), help="The nonsystematic encoder you wish to test against. The help menu will show the available options. See paper for more details.")
parser.add_argument('decoder', type=NonsystematicDecoders, choices=list(NonsystematicDecoders), help="The kind of BCJR decoder to use. 'basic' is a traditional BCJR decoder reworked for nonsystematic codes. This is the only type.")

parser.add_argument('--num_batches', type=int, default=1, help="Number of batches to sample and benchmark code on. `num_batches`*`batch_size` = total number of blocks tested.")
parser.add_argument('--block_len', type=int, default=100, help="Block length at which to run the code.")
parser.add_argument('--num_iter', type=int, default=6, help="Number of iterations to run BCJR decoder.")
parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for benchmarking. Larger batch uses more memory but can speed up benchmarking.`num_batches`*`batch_size` = total number of blocks tested.")

parser.add_argument('--snr_test_start', type=float, default=-1.5, help="The low point for sampling a range of SNRs to test at.")
parser.add_argument('--snr_test_end', type=float, default=2.0, help="The high point for sampling a range of SNRs to test at.")
parser.add_argument('--snr_points', type=int, default=8, help="Number of SNR testing points to be sampled. They will be evenly distributed along `snr_test_start` and `snr_test_end`.")

parser.add_argument('--channel', type=NoisyChannels, choices=list(NoisyChannels), default=NoisyChannels.AWGN, help="Noisy channel to test against. Default is AWGN.")

parser.add_argument('--write_logfile', action='store_true', help="This will write your results and many more details to a logfile in the test_logs folder. The logfile will be named based on the 'model-id' printed to stdout. It can easily be identified as it will have a timestamp in the name. It will automatically create the folder if there is none. These logfiles will store BER performance along with other helpful metrics.")
parser.add_argument('--write_tensorboard', action='store_true', help="This will write the results in the logfile to Tensorboard as well. You can then use Tensorboard to visualize those results.")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    snrs, test_sigmas = get_test_sigmas(
        args.snr_test_start, args.snr_test_end, args.snr_points)
    
    encoder = cast(NonsystematicTurboEncoderSpec, ENCODERS[args.encoder]())
    decoder_factory = DECODERS[args.decoder]
    channel = CHANNELS[args.channel]

    model_title = '_'.join([str(args.encoder), str(args.decoder)])
    encoder_decoders = [
        TurboNonsystematicEncoderDecoder(
            encoder.noninterleaved_code, 
            encoder.interleaved_code, 
            channel(sigma=sigma, block_len=args.block_len),
            decoder_factory, 
            args.block_len, 
            False, 
            args.num_iter,
            name=model_title
        ) for sigma in test_sigmas]
    
    assert all(ed.rate == (1, 3) for ed in encoder_decoders)

    results_writer = ResultsWriter(
        model_title=model_title, 
        # debugging
        # logdir='./tmp/test_logs/'
        logdir='./test_logs/'
    )

    validators = [
        Validator(
            enc_dec, 
            block_len=args.block_len, 
            batch_size=args.batch_size, 
            write_results_to_log=args.write_logfile,
            results_writer=results_writer,
            write_to_tensorboard=args.write_tensorboard,
            # debugging
            # tensorboard_dir='./tmp/tensorboard_dir/',
            tensorboard_dir='./tensorboard/testing/',
            verbose=1
        )
        for enc_dec in encoder_decoders
    ]

    steps = args.num_batches
    for i, validator in enumerate(validators, start=1):
        print(f'Running test {i}/{len(validators)}')
        pprint(asdict(validator.settings()))
        validator.run(steps, tb_step=i)